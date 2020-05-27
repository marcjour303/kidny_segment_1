import os

import nibabel as nib
import numpy as np
import torch
import json

import utils.common_utils as common_utils
import utils.data_utils as du

import utils.kits_data_utils as kutils
import matplotlib.pyplot as plt

def dice_confusion_matrix(vol_output, ground_truth, num_classes, no_samples=10, mode='train'):
    dice_cm = torch.zeros(num_classes, num_classes)
    if mode == 'train':
        samples = np.random.choice(len(vol_output), no_samples)
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        for j in range(num_classes):
            Pred = (vol_output == j).float()
            inter = torch.sum(torch.mul(GT, Pred))
            union = torch.sum(GT) + torch.sum(Pred) + 0.0001
            dice_cm[i, j] = 2 * torch.div(inter, union)
    avg_dice = torch.mean(torch.diagflat(dice_cm))
    return avg_dice, dice_cm


def dice_score_perclass(vol_output, ground_truth, num_classes, no_samples=10, mode='train'):
    dice_perclass = torch.zeros(num_classes)
    #if mode == 'train':
    #    samples = np.random.choice(len(vol_output), no_samples)
    #    vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    class_label = 1
    GT = (ground_truth == class_label).float()
    Pred = (vol_output == class_label).float()
    inter = torch.sum(GT * Pred)
    union = torch.sum(GT) + torch.sum(Pred) + 0.0001
    dice_score_kidney = (2 * torch.div(inter, union))
    return dice_score_kidney


def image_per_epoch(prediction):
    print("Sample Images...", end='', flush=True)
    ncols = 1
    nrows = len(prediction)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 20))
    for i in range(nrows):
        ax[i].imshow(prediction[i], cmap='CMRmap', vmin=0, vmax=1)
        ax[i].set_title("Predicted", fontsize=10, color="blue")
        ax[i].axis('off')
    fig.set_tight_layout(True)
    plt.show(fig)
    print('printed', flush=True)


def evaluate_dice_score(model, test_loader, model_path, num_classes, data_dir, volumes_txt_file, remap_config,
                        orientation,
                        prediction_path, device=0, logWriter=None, mode='eval', downsample=1):
    print("**Starting evaluation. Please check tensorboard for plots if a logWriter is provided in arguments**")

    batch_size = 2
    volumes_to_use = []

    print("Running on: ", device)
    print("Using model from: ", model_path)

    if not device:
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    else:
        # model.load_state_dict(torch.load(model_path, map_location=device))
        model = torch.load(model_path, map_location=device)
    print("The model is loaded successfuly")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        model.cuda(device)

    model.eval()

    common_utils.create_if_not(prediction_path)
    volume_dice_score_list = []
    print("Evaluating now...")
    eval_on_test = False
    eval_on_train = True
    eval_on_validate = False
    file_paths = []
    visu_dir = "eval"

    if eval_on_test:
        print("Loading test cases from : ", volumes_txt_file)
        try:
            with open(volumes_txt_file, 'r') as tst:
                test_case_ids = json.load(tst)
                volumes_to_use = test_case_ids['test_cases']
        except IOError:
            print("No evaluation data")
        print(volumes_to_use)
        vol_files = kutils.get_data_paths(data_dir, volumes_to_use)

        file_paths = [
            [os.path.join(vol, 'imaging.nii.gz'), os.path.join(vol, 'segmentation.nii.gz')]
            for
            vol in vol_files]

        visu_dir = "eval"
    elif eval_on_train:
        try:
            with open("E:\\quicknat-master\\datasets\\train_data.json", 'r') as tst:
                test_case_ids = json.load(tst)
                file_paths = test_case_ids['cases']
                n = len(file_paths)
                file_paths = file_paths[:int(n*0.8)]
                print("evaluating on training data")
        except IOError:
            print("No train data")
        visu_dir = "train"
    elif eval_on_validate:
        try:
            with open("E:\\quicknat-master\\datasets\\val_data.json", 'r') as tst:
                test_case_ids = json.load(tst)
                file_paths = test_case_ids['cases']
                file_paths = file_paths[:2]
                print("evaluating on training data")
        except IOError:
            print("No validation data")
        visu_dir = "validate"

    print(file_paths)
    with torch.no_grad():
        for vol_idx, file_path in enumerate(file_paths):
            volume_loaded, labelmap, class_weights, weights, header = du.load_and_preprocess(file_path,
                                                                                      orientation=orientation,
                                                                                      remap_config=remap_config,
                                                                                      downsample=downsample)

            volume = volume_loaded if len(volume_loaded.shape) == 4 else volume_loaded[:, np.newaxis, :, :]
            volume, labelmap = torch.tensor(volume).type(torch.FloatTensor), torch.tensor(labelmap).type(
                torch.LongTensor)

            volume_prediction = []
            print(len(volume))
            print(batch_size)
            for i in range(0, len(volume), batch_size):
                batch_x, batch_y = volume[i: i + batch_size], labelmap[i:i + batch_size]
                if cuda_available:
                    batch_x = batch_x.cuda(device)
                out = model(batch_x)

                _, batch_output = torch.max(out, dim=1)
                #print("batch output sum", torch.sum(batch_output))
                volume_prediction.append(batch_output)

            volume_prediction = torch.cat(volume_prediction)
            # volume_dice_score = dice_score_perclass(volume_prediction, labelmap.cuda(device), num_classes, mode=mode)

            volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')

            print("Input shape : ", volume.shape)
            print("The shape of the result: ", volume_prediction.shape)
            #nifti_img = nib.MGHImage(volume_prediction, np.eye(4), header=header)

            case_path, file_name = os.path.split(file_path[0])
            data_pt, case_name = os.path.split(case_path)

            #pred_file_path = os.path.join(prediction_path, case_name + str('_pred.nii'))
            #print(pred_file_path)
            #nib.save(nifti_img, pred_file_path)

            print("Estimate background vals: ", np.count_nonzero(volume_prediction == 0))
            print("Estimate kidney vals: ", np.count_nonzero(volume_prediction == 1))

            #vu.visualize(case_name, volume_loaded, volume_prediction, os.path.join(prediction_path, visu_dir))
            # if logWriter:
            #    logWriter.plot_dice_score('val', 'eval_dice_score', volume_dice_score, volumes_to_use[vol_idx], vol_idx)

            # volume_dice_score = volume_dice_score.cpu().numpy()
            # volume_dice_score_list.append(volume_dice_score)
            # print(volume_dice_score, np.mean(volume_dice_score))
        # dice_score_arr = np.asarray(volume_dice_score_list)
        # avg_dice_score = np.mean(dice_score_arr)
        # print("Mean of dice score : " + str(avg_dice_score))
        # class_dist = [dice_score_arr[:, c] for c in range(num_classes)]

        # if logWriter:
        #    logWriter.plot_eval_box_plot('eval_dice_score_box_plot', class_dist, 'Box plot Dice Score')
    print("DONE")

    return None, None
    # return avg_dice_score, class_dist
