import itertools
import logging
import os
import re
import shutil
import stat
from textwrap import wrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter

import utils.evaluator as eu

plt.switch_backend('agg')
plt.axis('scaled')


# TODO: Add custom phase names


def on_rm_error(func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod(path, stat.S_IWRITE)
    shutil.rmtree(path)


class LogWriter(object):
    def __init__(self, log_dir_name, exp_name, use_last_checkpoint=False, labels=None,
                 cm_cmap=plt.cm.Blues):
        self.num_class=1
        train_log_path, val_log_path, eval_log_path = os.path.join(log_dir_name, exp_name, "train"), \
                                                      os.path.join(log_dir_name,exp_name,"val"), \
                                                      os.path.join(log_dir_name, exp_name, "eval")

        if not use_last_checkpoint:
            if os.path.exists(train_log_path):
                shutil.rmtree(train_log_path, onerror=on_rm_error)
            if os.path.exists(val_log_path):
                shutil.rmtree(val_log_path, onerror=on_rm_error)
            if os.path.exists(eval_log_path):
                shutil.rmtree(eval_log_path, onerror=on_rm_error)

        self.writer = {
            'train': SummaryWriter(train_log_path),
            'val': SummaryWriter(val_log_path),
            'eval': SummaryWriter(eval_log_path)
        }
        self.curr_iter = 1
        self.cm_cmap = cm_cmap
        self.labels = self.beautify_labels(labels)
        self.logger = logging.getLogger()
        file_handler = logging.FileHandler("{0}/{1}.log".format(os.path.join(log_dir_name, exp_name), "console_logs"))
        self.logger.addHandler(file_handler)

    def log(self, text, phase='train'):
        self.logger.info(text)

    def loss_per_iter(self, acc_loss_val, dice_loss_value, ce_loss_value,  i_batch, phase, iteration):
        print('[Iteration : ' + str(i_batch) + '] CE Loss -> ' + str(ce_loss_value))
        print('[Iteration : ' + str(i_batch) + '] Dice Loss -> ' + str(dice_loss_value))
        self.writer[phase].add_scalar('acc_loss/per_iteration', acc_loss_val, iteration)
        self.writer[phase].add_scalar('dice_loss/per_iteration', dice_loss_value, iteration)
        self.writer[phase].add_scalar('ce_loss/per_iteration', ce_loss_value, iteration)

    def loss_per_epoch(self, loss_type, loss_arr, phase, epoch):
        loss = np.mean(loss_arr)
        self.writer[phase].add_scalar(loss_type + '/per_epoch', loss, epoch)
        print('epoch ' + phase + ' loss = ' + str(loss))

    def dice_score_per_epoch(self, phase, ds, epoch):
        self.writer[phase].add_scalar('dice_score_per_epoch', ds, epoch)
        return ds.item()

    def plot_dice_score(self, phase, caption, ds, title, step=None):
        fig = matplotlib.figure.Figure(figsize=(8, 6), dpi=180, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(title, fontsize=10)
        ax.xaxis.set_label_position('top')
        ax.bar(np.arange(self.num_class), ds)
        ax.set_xticks(np.arange(self.num_class))
        c = ax.set_xticklabels(self.labels, fontsize=6, rotation=-90, ha='center')
        ax.xaxis.tick_bottom()
        if step:
            self.writer[phase].add_figure(caption + '/' + phase, fig, step)
        else:
            self.writer[phase].add_figure(caption + '/' + phase, fig)

    def plot_eval_box_plot(self, caption, class_dist, title):
        fig = matplotlib.figure.Figure(figsize=(8, 6), dpi=180, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(title, fontsize=10)
        ax.xaxis.set_label_position('top')
        ax.boxplot(class_dist)
        ax.set_xticks(np.arange(self.num_class))
        c = ax.set_xticklabels(self.labels, fontsize=6, rotation=-90, ha='center')
        ax.xaxis.tick_bottom()
        self.writer['val'].add_figure(caption, fig)

    def image_per_epoch(self, predictions, labels, phase, epoch):
        ncols = 2
        nrows = len(predictions)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 20))
        for row_idx in range(nrows):
            prediction = predictions[row_idx]
            ground_truth = labels[row_idx]
            ax[row_idx, 0].imshow(prediction, vmax=abs(prediction).max(), vmin=-abs(prediction).max())
            ax[row_idx, 0].set_title("Predicted", fontsize=10, color="blue")
            ax[row_idx, 0].axis('off')
            ax[row_idx, 1].imshow(ground_truth, vmax=abs(ground_truth).max(), vmin=-abs(ground_truth).max())
            ax[row_idx, 1].set_title("Ground Truth", fontsize=10, color="blue")
            ax[row_idx, 1].axis('off')
            fig.set_tight_layout(True)

        self.writer[phase].add_figure('sample_prediction/' + phase, fig, epoch)


    def graph(self, model, X):
        self.writer['train'].add_graph(model, X)

    def close(self):
        self.writer['train'].close()
        self.writer['val'].close()
        self.writer['eval'].close()

    def beautify_labels(self, labels):
        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
        classes = ['\n'.join(wrap(l, 40)) for l in classes]
        return classes
