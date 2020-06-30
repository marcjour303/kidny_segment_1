import argparse
import os

import torch
from settings import Settings as stng

import utils.evaluator as eu

from quicknat import QuickNat
from solver import Solver

#utility imports
import utils.data_utils as du
from utils.data_loader import ToTensor, NiftiData
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

    du.filter_and_split_data(data_params)
    train_files = du.load_volume_paths_from_case_file(data_params["data_dir"], data_params["train_data_file"])
    val_files = du.load_volume_paths_from_case_file(data_params["data_dir"], data_params["val_data_file"])

    train_data = NiftiData(train_files, data_params, mode='train')
    val_data = NiftiData(val_files, data_params, mode='val')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_params['batch_step_size'], shuffle=True,
                                               num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=train_params['batch_step_size'], shuffle=True,
                                             num_workers=4, pin_memory=True)

    if train_params['use_pre_trained']:
        quicknat_model = QuickNat(net_params)
        quicknat_model.load_state_dict(torch.load(train_params['pre_trained_path']))
    else:
        quicknat_model = QuickNat(net_params)

    solver = Solver(quicknat_model,
                    exp_name=train_params['exp_name'],
                    device=common_params['device'],
                    num_class=net_params['num_class'],
                    optim_args={"lr": train_params['learning_rate'],
                                "betas": train_params['optim_betas'],
                                "eps": train_params['optim_eps'],
                                "weight_decay": train_params['optim_weight_decay']},
                    model_name=common_params['model_name'],
                    labels=data_params['labels'],
                    log_nth=train_params['log_nth'],
                    num_epochs=train_params['num_epochs'],
                    lr_scheduler_step_size=train_params['lr_scheduler_step_size'],
                    lr_scheduler_gamma=train_params['lr_scheduler_gamma'],
                    use_last_checkpoint=train_params['use_last_checkpoint'],
                    log_dir=common_params['log_dir'],
                    exp_dir=common_params['exp_dir'],
                    train_batch_size=train_params['train_batch_size'],
                    val_batch_size=train_params['val_batch_size'])

    solver.train(train_loader, val_loader)

    final_model_path = os.path.join(common_params['save_model_dir'], train_params['final_model_file'])
    solver.save_best_model(final_model_path)

    print("final model saved @ " + str(final_model_path))



def evaluate(eval_params, net_params, data_params, common_params, train_params):

    eval_model_path = eval_params['eval_model_path']

    save_predictions_dir = eval_params['save_predictions_dir']
    # go to evaluator, remap_labels and add an option "do nothing", because you don't need to remap anything

    device = common_params['device']
    log_dir = common_params['log_dir']
    exp_dir = common_params['exp_dir']
    exp_name = train_params['exp_name']
    prediction_path = os.path.join(exp_dir, exp_name, save_predictions_dir)

    print("Loading pretrained model")
    #Load trained model
    print(eval_params['eval_model_path'])
    model_path = eval_params['eval_model_path']
    quicknat_model= QuickNat(net_params)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        quicknat_model.cuda(device)

    if not device:
        quicknat_model.load_state_dict(torch.load(model_path))
        quicknat_model.to(device)
    else:
        #checkpoint = torch.load(model_path)
        #quicknat_model.load_state_dict(checkpoint['state_dict'])
        #quicknat_model.load_state_dict(torch.load(model_path, map_location=device))

        quicknat_model = torch.load(eval_params['eval_model_path'],  map_location = device)

    # Load test data
    print("Loading test data")
    eval_files = du.load_volume_paths_from_case_file(data_params["data_dir"], data_params["val_data_file"])
    eval_data = NiftiData(eval_files[:1], data_params, mode='eval')
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=train_params['batch_step_size'],
                                              shuffle=False, num_workers=4, pin_memory=True)

    #Evaluate and log model results
    logWriter = LogWriter(log_dir, exp_name, labels=data_params['labels'])
    dice_score = eu.evaluate_dice_score(model=quicknat_model,
                                        data_loader=eval_loader,
                                        device=device,
                                        logWriter=logWriter)
    print("Average test dice score: ", dice_score)
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
    print("Running with configuration:")
    print("COMMON_PARAMS")
    print(common_params)
    print("NET_PARAMS")
    print(net_params)
    print("DATA_PARAMS")
    print(data_params)
    print("TRAIN_PARAMS")
    print(train_params)
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
