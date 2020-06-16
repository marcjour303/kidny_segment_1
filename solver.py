import glob
import os

import numpy as np
import torch
# from nn_common_modules import losses as additional_losses
import losses as additional_losses
from torch.optim import lr_scheduler

import utils.common_utils as common_utils
from utils.log_utils import LogWriter
import utils.evaluator as eu
import matplotlib.pyplot as plt
from pathlib import Path
import random
import polyaxon_helper
from torch.autograd import Variable

CHECKPOINT_DIR = "tst"  # os.path.join(polyaxon_helper.get_outputs_path(), 'checkpoints')

# CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_EXTENSION = 'pth.tar'


class Solver(object):

    def __init__(self,
                 model,
                 exp_name,
                 device,
                 num_class,
                 optim=torch.optim.Adam,
                 optim_args={},
                 loss_func=additional_losses.CombinedLoss(),
                 model_name='quicknat',
                 labels=None,
                 num_epochs=10,
                 log_nth=5,
                 lr_scheduler_step_size=5,
                 lr_scheduler_gamma=0.5,
                 use_last_checkpoint=True,
                 exp_dir='experiments',
                 log_dir='logs',
                 train_batch_size=8,
                 val_batch_size=8):

        self.device = device
        self.model = model

        self.model_name = model_name
        self.labels = labels
        self.num_epochs = num_epochs
        if torch.cuda.is_available():
            self.loss_func = loss_func.cuda(device)
        else:
            self.loss_func = loss_func

        print("Optimization arguments: ", optim_args)
        self.optim = optim(model.parameters(), **optim_args)
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=lr_scheduler_step_size,
                                             gamma=lr_scheduler_gamma)

        #self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, patience=2, mode='min')
        exp_dir_path = os.path.join(exp_dir, exp_name)
        common_utils.create_if_not(exp_dir_path)
        common_utils.create_if_not(os.path.join(exp_dir_path, CHECKPOINT_DIR))
        self.exp_dir_path = exp_dir_path

        self.log_nth = log_nth
        print(log_dir)
        self.logWriter = LogWriter(num_class, log_dir, exp_name, use_last_checkpoint, labels)

        self.use_last_checkpoint = use_last_checkpoint

        self.start_epoch = 1
        self.start_iteration = 1

        self.best_ds_mean = 0
        self.best_ds_mean_epoch = 0

        self.val_batch_size = val_batch_size
        self.train_batch_size = train_batch_size

        if use_last_checkpoint:
            self.load_checkpoint()

    # TODO:Need to correct the CM and dice score calculation.
    #@profile
    def train(self, train_loader, val_loader):
        """
        Train a given model with the provided data.

        Inputs:
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        """
        model, optim, scheduler = self.model, self.optim, self.scheduler

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model.cuda(self.device)

        print('START TRAINING. : model name = %s, device = %s' % (
            self.model_name, torch.cuda.get_device_name(self.device)))
        train_iteration = self.start_iteration
        val_iteration = self.start_iteration
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            print("\n==== Epoch [ %d  /  %d ] START ====" % (epoch, self.num_epochs))

            self.model.train()
            phase = 'train'
            train_iteration = self.train_batch(train_loader, epoch, phase, train_iteration)
            with torch.no_grad():
                self.model.eval()
                phase = 'val'
                val_iteration = self.train_batch(val_loader, epoch, phase, val_iteration)

            print("==== Epoch [" + str(epoch) + " / " + str(self.num_epochs) + "] DONE ====")
            self.save_checkpoint({
                'epoch': epoch + 1,
                'start_iteration': train_iteration + 1,
                'arch': self.model_name,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(self.exp_dir_path, CHECKPOINT_DIR,
                            'checkpoint_epoch_' + str(epoch) + '.' + CHECKPOINT_EXTENSION))

        print('FINISH.')

        self.logWriter.close()

    def train_batch(self, data_loader, epoch, phase, iteration):

        loss_arr = []

        print("<<<= Phase: %s =>>>" % phase)
        for i_batch, sample_batched in enumerate(data_loader):
            model = self.model
            X = sample_batched[0].type(torch.FloatTensor)
            y = sample_batched[1].type(torch.LongTensor)
            class_w = sample_batched[2].type(torch.FloatTensor)

            if model.is_cuda:
                X, y, class_w = X.cuda(self.device, non_blocking=True), \
                                   y.cuda(self.device, non_blocking=True), \
                                   class_w.cuda(self.device, non_blocking=True)

            output = model(X)
            loss = self.loss_func(output, y, class_weight=class_w)

            with torch.no_grad():
                dice, ce = additional_losses.log_losses(output.detach(), y.detach(), class_w.detach())

            if i_batch % self.log_nth == 0:
                self.logWriter.loss_per_iter(loss.item(), dice, ce, i_batch, phase, iteration)

            loss_arr.append(loss.item())

            if phase == 'train':
                loss.backward()
                if (i_batch + 1) % self.train_batch_size == 0:
                    self.optim.step()
                    self.scheduler.step()
                    self.optim.zero_grad()

            del X, y, class_w, output, loss
            del model
            torch.cuda.empty_cache()
            iteration += 1
        self.log_results(data_loader, loss_arr, phase, epoch)
        return iteration

    def val_batch(self, data_loader, epoch, iteration):
        self.model.eval()
        phase = 'val'
        loss_arr = []
        print("<<<= Phase: %s =>>>" % phase)

        for i_batch, sample_batched in enumerate(data_loader):
            model = self.model
            X = sample_batched[0].type(torch.FloatTensor)
            y = sample_batched[1].type(torch.LongTensor)
            class_w = sample_batched[2].type(torch.FloatTensor)
            if model.is_cuda:
                X, y, class_w = X.cuda(self.device, non_blocking=True), \
                                   y.cuda(self.device, non_blocking=True), \
                                   class_w.cuda(self.device, non_blocking=True)
            output = model(X)
            loss = self.loss_func(output, y, class_weight=class_w)
            loss_arr.append(loss.item())

            dice, ce = additional_losses.log_losses(output.detach(), y.detach(), class_w.detach())
            if i_batch % self.log_nth == 0:
                self.logWriter.loss_per_iter(loss.item(), dice, ce, i_batch, phase, iteration)

            del X, y, class_w, output, loss
            del model
            torch.cuda.empty_cache()
            iteration += 1

        self.log_results(data_loader, loss_arr, phase, epoch)
        return iteration

    def log_results(self, data_loader, loss_arr, phase, epoch):
        print("Computing model accuracy ")
        with torch.no_grad():
            ds = 0
            batch_count = 0
            self.logWriter.loss_per_epoch(loss_arr, phase, epoch)
            for i_batch, sample_batched in enumerate(data_loader):
                x_in = sample_batched[0].type(torch.FloatTensor)
                labels = sample_batched[1].type(torch.LongTensor).squeeze()
                if self.model.is_cuda:
                    x_in, labels = x_in.cuda(self.device, non_blocking=True),\
                                   labels.cuda(self.device, non_blocking=True)

                pred = self.model.predict(x_in, self.device)
                ds += eu.dice_score_perclass(pred, labels)
                batch_count += 1
                print(".", end="", flush=True)

            ds = ds / batch_count
            print("", flush=True)
            print("Dice score ", ds)
            self.logWriter.dice_score_per_epoch(phase, ds, epoch)

            if phase == 'val' and ds > self.best_ds_mean:
                self.best_ds_mean = ds
                self.best_ds_mean_epoch = epoch

    def save_best_model(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".
        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        self.load_checkpoint(self.best_ds_mean_epoch)

        torch.save(self.model, path)

    def save_checkpoint(self, state, filename):
        print('saving model:', filename)
        torch.save(state, filename)

    def load_checkpoint(self, epoch=None):
        if epoch is not None:
            checkpoint_path = os.path.join(self.exp_dir_path, CHECKPOINT_DIR,
                                           'checkpoint_epoch_' + str(epoch) + '.' + CHECKPOINT_EXTENSION)
            self._load_checkpoint_file(checkpoint_path)
        else:
            all_files_path = os.path.join(self.exp_dir_path, CHECKPOINT_DIR, '*.' + CHECKPOINT_EXTENSION)
            list_of_files = glob.glob(all_files_path)
            if len(list_of_files) > 0:
                checkpoint_path = max(list_of_files, key=os.path.getctime)
                self._load_checkpoint_file(checkpoint_path)
            else:
                self.logWriter.log(
                    "=> no checkpoint found at '{}' folder".format(os.path.join(self.exp_dir_path, CHECKPOINT_DIR)))

    def _load_checkpoint_file(self, file_path):
        self.logWriter.log("=> loading checkpoint '{}'".format(file_path))
        checkpoint = torch.load(file_path)
        self.start_epoch = checkpoint['epoch']
        self.start_iteration = checkpoint['start_iteration']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer'])

        for state in self.optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.logWriter.log("=> loaded checkpoint '{}' (epoch {})".format(file_path, checkpoint['epoch']))

    def log_best_model_results(self, val_data):

        self.load_checkpoint(self.best_ds_mean_epoch)
        self.model.eval()
        index = np.random.choice(len(val_data), 5, replace=False)
        for idx in index:
            # print("Logging image with index: ", idx)
            v_img, v_label, _, _ = val_data.dataset[idx]
            prediction = self.model.predict(v_img.unsqueeze(dim=0), self.device)
            torch.cuda.empty_cache()
            self.logWriter.best_model_validation_images(prediction, v_label)

        print('FINISH.')

        self.logWriter.close()
