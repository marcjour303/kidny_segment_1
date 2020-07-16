import os
import torch
import matplotlib.pyplot as plt
import utils.data_loader as du
from pathlib import Path
import numpy as np


def dice_confusion_matrix(vol_output, ground_truth, num_classes, mode='train'):
    dice_cm = torch.zeros(num_classes, num_classes)
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        for j in range(num_classes):
            Pred = (vol_output == j).float()
            inter = torch.sum(torch.mul(GT, Pred))
            union = torch.sum(GT) + torch.sum(Pred) + 0.0001
            dice_cm[i, j] = 2 * torch.div(inter, union)
    avg_dice = torch.mean(torch.diagflat(dice_cm))
    return avg_dice, dice_cm


def dice_score_perclass(vol_output, ground_truth, print_debug = False):

    unique, counts = torch.unique(ground_truth, return_counts=True)

    smooth = 1
    dice_score_kidney = [-1, -1]
    for idx, label in enumerate(unique):
        GT = (ground_truth == label).float()
        Pred = (vol_output == label).float()
        inter = torch.sum(GT * Pred)
        union = torch.sum(GT) + torch.sum(Pred)

        dice_score_kidney[idx] = (2 * torch.div(inter + smooth, union + smooth))
    return dice_score_kidney


def kidney_dice_score(vol_output, ground_truth, print_debug = False):

    kidney_label = 1
    smooth = 1

    GT = (ground_truth == kidney_label).float()
    Pred = (vol_output == kidney_label).float()
    inter = 2 * torch.sum(GT * Pred)
    union = torch.sum(GT) + torch.sum(Pred)

    dice_score_kidney = torch.div(inter + smooth, union + smooth)

    return dice_score_kidney


def eval_results_per_epoch(model, device, data_loader, log_writer, phase, epoch):
        print("Computing model accuracy ")
        with torch.no_grad():
            ds = 0
            vol_numbers = data_loader.dataset.get_volume_numbers()
            best_batch_ds = -1
            worst_batch_ds = -1

            best_batch = {}
            worst_batch = {}

            for vol_idx in range(vol_numbers):
                vol_file, label_file = data_loader.dataset.get_volume_file_name(vol_idx)
                vol_loader = du.get_volume_mini_dataloader(vol_file, label_file, 8)

                predicted_label = torch.tensor(np.array([])).cuda(device, non_blocking=True)
                gt_label = torch.tensor(np.load(label_file)).cuda(device, non_blocking=True)
                for i_batch, sample_batched in enumerate(vol_loader):
                    x_in = sample_batched[0].type(torch.FloatTensor)
                    lb = sample_batched[1].type(torch.FloatTensor)
                    if model.is_cuda:
                        x_in = x_in.cuda(device, non_blocking=True)
                        lb = lb.cuda(device, non_blocking=True)

                    pred = model.predict(x_in, device)

                    lb = lb.squeeze(axis=1)
                    batch_ds = kidney_dice_score(pred, lb)

                    if torch.sum(lb) != 0:

                        if worst_batch_ds == -1 or batch_ds < worst_batch_ds:
                            worst_batch['pred'] = pred.cpu().numpy()
                            worst_batch['gt'] = lb.cpu().numpy()
                            worst_batch['input'] = x_in.squeeze(axis=1).cpu().numpy()
                            worst_batch['score'] = batch_ds
                            worst_batch_ds = batch_ds

                        if best_batch_ds == -1 or batch_ds > best_batch_ds:
                            best_batch['pred'] = pred.cpu().numpy()
                            best_batch['gt'] = lb.cpu().numpy()
                            best_batch['input'] = x_in.squeeze(axis=1).cpu().numpy()
                            best_batch['score'] = batch_ds
                            best_batch_ds = batch_ds

                    if i_batch == 0:
                        predicted_label = pred
                    else:
                        predicted_label = torch.cat([predicted_label, pred], dim=0)

                ds += kidney_dice_score(predicted_label, gt_label)

            ds = ds / vol_numbers
            print("", flush=True)
            print("Dice score ", ds)
            log_writer.dice_score_per_epoch(phase, ds, epoch)
            log_writer.image_per_epoch(batch_data=best_batch, batch_type='best_batch', phase=phase, epoch=epoch)
            log_writer.image_per_epoch(batch_data=worst_batch, batch_type='worst_batch', phase=phase, epoch=epoch)

        return ds


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


def evaluate_dice_score(model, data_loader, device=0, logWriter=None, phase='eval' ):
    print("**Starting evaluation. Please check tensorboard for plots if a logWriter is provided in arguments**")
    print("Running on: ", device)
    print("The model is loaded successfuly")

    model.eval()
    print("<<<= Phase: %s =>>>" % phase)

    with torch.no_grad():
        avg_dice_score = eval_results_per_epoch(model, device, data_loader, logWriter, phase, 1)

    print("DONE")

    return avg_dice_score
