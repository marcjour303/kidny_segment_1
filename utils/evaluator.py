import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def dice_score_perclass(vol_output, ground_truth, print_debug = False):

    unique, counts = torch.unique(ground_truth, return_counts=True)
    inter = 0
    union = 0
    eps = 0.00001
    for idx, label in enumerate(unique):
        GT = (ground_truth == label).float()
        Pred = (vol_output == label).float()
        inter += (1.0 / (counts[idx]**2 + eps)) * torch.sum(GT * Pred)
        union += (1.0 / (counts[idx]**2 + eps)) * (torch.sum(GT) + torch.sum(Pred))

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

def evaluate_dice_score(model, data_loader, device=0, logWriter=None, phase='eval' ):
    print("**Starting evaluation. Please check tensorboard for plots if a logWriter is provided in arguments**")

    print("Running on: ", device)

    #if not device:
    #    model.to(device)

    print("The model is loaded successfuly")

    #if torch.cuda.is_available():
    #    torch.cuda.empty_cache()
    #    model.cuda(device)

    model.eval()

    print("<<<= Phase: %s =>>>" % phase)
    ds = 0
    batch_count = 0
    batch_min_ds = 1
    batch_max_ds = 0
    debug_path = os.path.join("E:\\", "vis_data")
    print("The debug path: ", str(debug_path))
    if not Path(debug_path).exists():
        Path(debug_path).mkdir()


    with torch.no_grad():
        for i_batch, sample_batched in enumerate(data_loader):

            x_in = sample_batched[0].type(torch.FloatTensor)
            labels = sample_batched[1].type(torch.LongTensor)

            if model.is_cuda:
                x_in, labels = x_in.cuda(device, non_blocking=True),\
                               labels.cuda(device, non_blocking=True)

            pred = model.predict(x_in, device)

            batch_ds = dice_score_perclass(pred.cpu(), labels.squeeze().cpu()) #, i_batch % 10 == 0)
            ds += batch_ds
            batch_count += 1

            if i_batch % 10 == 0:
                vis_pred = pred.cpu()
                vis_label = labels.squeeze().cpu()

                #logWriter.image_per_epoch(vis_pred, vis_label, phase, 1)
                b_size = len(x_in)
                fig, axs = plt.subplots(nrows=3, ncols=b_size, figsize=(3, 5)) #sharex='all'
                for b_i in range(b_size):

                    axs[0, b_i].imshow(x_in.cpu().squeeze()[b_i, :, :])
                    axs[1, b_i].imshow(vis_label[b_i, :, :])
                    axs[2, b_i].imshow(vis_pred[b_i, :, :])
                fig_name = os.path.join(debug_path, str(i_batch) + '.png')
                fig.savefig(fig_name)

                print("Batch ", str(i_batch), " ds: ", batch_ds, flush=True)



            del x_in, labels, pred
            torch.cuda.empty_cache()

    avg_dice_score = ds / batch_count

            # if logWriter:
            #    logWriter.plot_dice_score('val', 'eval_dice_score', volume_dice_score, volumes_to_use[vol_idx], vol_idx)

            # volume_dice_score = volume_dice_score.cpu().numpy()
            # volume_dice_score_list.append(volume_dice_score)
            # print(volume_dice_score, np.mean(volume_dice_score))

        # if logWriter:
        #    logWriter.plot_eval_box_plot('eval_dice_score_box_plot', class_dist, 'Box plot Dice Score')
    print("DONE")

    return avg_dice_score
