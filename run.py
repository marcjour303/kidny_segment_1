import argparse
import os

import torch
from settings import Settings as stng

import utils.evaluator as eu

from quicknat import QuickNat
from solver import Solver

#utility imports
import utils.data_utils as du
from utils.data_loader import DataloaderNII, ToTensor, NiftiData
from utils.log_utils import LogWriter
import shutil
import ast
from torchvision import transforms


from polyaxon_client.tracking import Experiment, get_data_paths
import polyaxon_helper

torch.set_default_tensor_type('torch.FloatTensor')


transform_train = transforms.Compose([
    ToTensor(),
])

transform_val = transforms.Compose([
    ToTensor(),
])


def train(train_params, common_params, data_params, net_params):
    train_files, val_files = du.apply_split(data_params["data_skip"], data_params["train_data_file"], data_params["val_data_file"],
                                            data_params["data_split"], data_params["data_dir"])

    train_ratio, test_ratio = data_params["data_split"].split(",")
    tt_ratio = float(test_ratio) / float(train_ratio)
    train_files = train_files[:10]
    h5_nr = 10
    iter_number = len(train_files) // h5_nr
    idx = 0
    idx_val = 0

    if False:
        for i in range(iter_number):
            c_t_d = train_files[idx:(idx + h5_nr)]
            c_v_d = val_files[idx_val:int((idx_val + h5_nr) * tt_ratio)]
            idx = idx + h5_nr
            idx_val = idx_val + int((idx_val + h5_nr) * tt_ratio)
            print("idx: " , idx)
            print("idx_val: ", idx_val)
            train_data = DataloaderNII(c_t_d, data_params, 'train', transform_train)
            val_data = DataloaderNII(c_v_d, data_params, 'val', transform_val)

            train_stage(train_data, val_data, train_params, common_params, data_params, net_params)
            use_pre_trained = True

        if len(train_files)% h5_nr != 0:
            c_t_d = train_files[idx:]
            c_v_d = val_files[idx_val:]

            train_data = DataloaderNII(c_t_d, data_params, 'train', transform_train)
            val_data = DataloaderNII(c_v_d, data_params, 'val', transform_val)

            train_stage(train_data, val_data, train_params, common_params, data_params, net_params)
    else:
        train_data = NiftiData(train_files[:1], data_params, train=True)
        val_data = NiftiData(val_files[:1], data_params, train=False)
        train_stage(train_data, val_data, train_params, common_params, data_params, net_params)


def train_stage(train_data, val_data, train_params, common_params, data_params, net_params):

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_params['train_batch_size'], shuffle=True,
                                               num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=train_params['val_batch_size'], shuffle=False,
                                             num_workers=4, pin_memory=True)

    if train_params['use_pre_trained']:
        quicknat_model = QuickNat(net_params)
        quicknat_model.load_state_dict(torch.load(train_params['pre_trained_path']))
    else:
        quicknat_model = QuickNat(net_params)

    solver = Solver(quicknat_model,
                    device=common_params['device'],
                    num_class=net_params['num_class'],
                    optim_args={"lr": train_params['learning_rate'],
                                "betas": train_params['optim_betas'],
                                "eps": train_params['optim_eps'],
                                "weight_decay": train_params['optim_weight_decay']},
                    model_name=common_params['model_name'],
                    exp_name=train_params['exp_name'],
                    labels=data_params['labels'],
                    log_nth=train_params['log_nth'],
                    num_epochs=train_params['num_epochs'],
                    lr_scheduler_step_size=train_params['lr_scheduler_step_size'],
                    lr_scheduler_gamma=train_params['lr_scheduler_gamma'],
                    use_last_checkpoint=train_params['use_last_checkpoint'],
                    log_dir=common_params['log_dir'],
                    exp_dir=common_params['exp_dir'])

    solver.train(train_loader, val_loader)
    #train_data.close_files()
    #val_data.close_files()
    final_model_path = os.path.join(common_params['save_model_dir'], train_params['final_model_file'])
    quicknat_model.save(final_model_path)
    print("final model saved @ " + str(final_model_path))


def evaluate(eval_params, net_params, data_params, common_params, train_params):

    eval_model_path = eval_params['eval_model_path']
    num_classes = net_params['num_class']
    labels = data_params['labels']
    data_dir = eval_params['data_dir']
    volumes_txt_file = eval_params['volumes_txt_file']
    # go to evaluator, remap_labels and add an option "do nothing", because you don't need to remap anything
    remap_config = eval_params['remap_config']
    device = common_params['device']
    log_dir = common_params['log_dir']
    exp_dir = common_params['exp_dir']
    exp_name = train_params['exp_name']
    save_predictions_dir = eval_params['save_predictions_dir']
    #prediction_path = os.path.join(exp_dir, exp_name, save_predictions_dir)
    prediction_path = os.path.join("E:\\", "CT_VIZ")
    orientation = eval_params['orientation']

    quicknat_model = QuickNat(net_params)

    logWriter = LogWriter(num_classes, log_dir, exp_name, labels=labels)

    dice_score = eu.evaluate_dice_score(quicknat_model, None, eval_model_path, num_classes, data_dir, volumes_txt_file,
                                        remap_config, orientation, prediction_path, device, logWriter, downsample=4)
    score = dice_score
    avg_dice_score, class_dist = score
    logWriter.close()


def delete_contents(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, help='run mode, valid values are train and eval')
    parser.add_argument('--settings', '-s', required=True, help='which settings file to use, valid values are local and cluster')
    parser.add_argument('--param_name', '-pm', required=False, help='learning rate for experiment groups', default=None)
    parser.add_argument('--param_value', '-pv', required=False, help='learning rate for experiment groups', default=None)
    args = parser.parse_args()


    if args.settings == 'local':
        print("running local config");
        settings_file = 'E:\quicknat-master\settings_local.ini'
    elif args.settings == 'cluster':
        settings_file = 'settings.ini'

    settings_dictionary = stng(settings_file).settings_dict
    common_params, data_params, net_params, train_params, eval_params = settings_dictionary['COMMON'], settings_dictionary['DATA'], settings_dictionary[
        'NETWORK'], settings_dictionary['TRAINING'], settings_dictionary['EVAL']

    if args.settings == 'cluster':
        # override some of the common_params in order to get the correct polyaxon paths
        common_params['log_dir'] = polyaxon_helper.get_outputs_path()
        common_params['save_model_dir'] = polyaxon_helper.get_outputs_path()
        common_params['exp_dir'] = polyaxon_helper.get_outputs_path()

    # override training vaues for experimant groups
    if args.param_name and args.param_value:
        print("Before: ", train_params[args.param_name])
        print("Adjusting experiment to run with learning rate: ", ast.literal_eval(args.param_value))
        train_params[args.param_name] = ast.literal_eval(args.param_value)
        print("After: ", train_params[args.param_name])

    if args.mode == 'train':
        train(train_params, common_params, data_params, net_params)
    elif args.mode == 'eval':
        evaluate(eval_params, net_params, data_params, common_params, train_params)
    elif args.mode == 'clear':
        shutil.rmtree(os.path.join(common_params['exp_dir'], train_params['exp_name']))
        print("Cleared current experiment directory successfully!!")
        shutil.rmtree(os.path.join(common_params['log_dir'], train_params['exp_name']))
        print("Cleared current log directory successfully!!")

    elif args.mode == 'clear-all':
        delete_contents(common_params['exp_dir'])
        print("Cleared experiments directory successfully!!")
        delete_contents(common_params['log_dir'])
        print("Cleared logs directory successfully!!")
    else:
        raise ValueError('Invalid value for mode. only support values are train, eval and clear')
