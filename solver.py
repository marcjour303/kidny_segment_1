import glob
import os

import numpy as np
import torch
# from nn_common_modules import losses as additional_losses
import losses as additional_losses
from torch.optim import lr_scheduler

import utils.common_utils as common_utils
from utils.log_utils import LogWriter

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
                 log_dir='logs'):

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
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model.cuda(self.device)

        print('START TRAINING. : model name = %s, device = %s' % (
            self.model_name, torch.cuda.get_device_name(self.device)))
        current_iteration = self.start_iteration

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            print("\n==== Epoch [ %d  /  %d ] START ====" % (epoch, self.num_epochs))
            for phase in ['train', 'val']:
                print("<<<= Phase: %s =>>>" % phase)
                loss_arr = []
                out_list = []
                y_list = []
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                for i_batch, sample_batched in enumerate(dataloaders[phase]):

                    X = sample_batched[0].type(torch.FloatTensor)
                    y = sample_batched[1].type(torch.LongTensor)
                    w = sample_batched[2].type(torch.FloatTensor)

                    if model.is_cuda:
                        X, y, w = X.cuda(self.device, non_blocking=True), \
                                  y.cuda(self.device, non_blocking=True), \
                                  w.cuda(self.device, non_blocking=True)

                    output = model(X)

                    #loss = self.loss_func(output, y, binary=True)
                    loss  = self.loss_func(output, y, w)
                    if phase == 'train':
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        scheduler.step()
                        if i_batch % self.log_nth == 0:
                            self.logWriter.loss_per_iter(loss.item(), i_batch, current_iteration)
                        current_iteration += 1

                    loss_arr.append(loss.item())

                    _, batch_output = torch.max(output, dim=1)
                    out_list.append(batch_output.cpu())
                    y_list.append(y.cpu())

                    del X, y, w, output, batch_output, loss
                    torch.cuda.empty_cache()
                    if phase == 'val':
                        if i_batch != len(dataloaders[phase]) - 1:
                            print("#", end='', flush=True)
                        else:
                            print("100%", flush=True)

                #out_path = Path(os.path.join("E:\\", "label_vis_data_utils", "predictions", phase))
                #if not out_path.exists():
                #    out_path.mkdir()

                with torch.no_grad():
                    out_arr, y_arr = torch.cat(out_list), torch.cat(y_list)
                    self.logWriter.loss_per_epoch(loss_arr, phase, epoch)
                    data_len = len(dataloaders[phase].dataset)
                    index = np.random.choice(data_len, 5, replace=False)
                    for idx in index:
                        #print("Logging image with index: ", idx)
                        v_img, v_label, w, _ = dataloaders[phase].dataset[idx]
                        name_val = random.randint(1, 100)
                        self.logWriter.image_per_epoch(model.predict(v_img.unsqueeze(dim=0), self.device), v_label, phase, epoch)
                        prediction = model.predict(v_img.unsqueeze(dim=0), self.device)
                        #fig, ax = plt.subplots(1, 4)
                        #_ = ax[0].imshow(v_label, cmap='Greys', vmax=abs(v_label).max(), vmin=abs(v_label).min())
                        #_ = ax[1].imshow(v_img.squeeze())
                        #_ = ax[2].imshow(w)
                        #_ = ax[3].imshow(prediction)
                        #fig_path = os.path.join(out_path, str(epoch) + "_" + "_img_" + str(name_val) + "_" + str(phase) + '.jpeg')
                        #print(str(fig_path))
                        #fig.savefig(str(fig_path))

                    self.logWriter.cm_per_epoch(phase, out_arr, y_arr, epoch)
                    ds_mean = self.logWriter.dice_score_per_epoch(phase, out_arr, y_arr, epoch)
                    print("Dice score per epoch: ", ds_mean)
                    if phase == 'val':
                        if ds_mean > self.best_ds_mean:
                            self.best_ds_mean = ds_mean
                            self.best_ds_mean_epoch = epoch

            print("==== Epoch [" + str(epoch) + " / " + str(self.num_epochs) + "] DONE ====")
            self.save_checkpoint({
                'epoch': epoch + 1,
                'start_iteration': current_iteration + 1,
                'arch': self.model_name,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(self.exp_dir_path, CHECKPOINT_DIR,
                            'checkpoint_epoch_' + str(epoch) + '.' + CHECKPOINT_EXTENSION))

        print('FINISH.')
        self.logWriter.close()

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
        print('saving mooodel:', filename)
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
