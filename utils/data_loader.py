import os

import torch
from torch.utils import data
import utils.preprocessor as preprocessor
import utils.data_utils as du

from pathlib import Path
import numpy as np
import nibabel as nb


class ToTensor(object):
    def __call__(self, sample):
        image, labels, class_weights = sample['image'], sample['label'], sample['class_weight']
        return {'image': torch.from_numpy(image.copy()),
                'label': torch.from_numpy(labels.copy()),
                'class_weight': torch.from_numpy(labels.copy())}


def norm(ar):
    ar = ar - np.min(ar)
    ar = ar / np.ptp(ar)
    return ar


class NiftiData(data.Dataset):

    def __init__(self, filePaths, data_params, mode='train'):
        self.mode = mode
        file_list = []
        for vol in filePaths:
            path_to_files = os.path.split(vol[0])[0]
            case_id = os.path.split(path_to_files)[1]
            file_list.append([vol[0], vol[1], case_id + '_class_weights.npy', case_id + '_weights.npy', 0, case_id])

        pointer = 0
        temp_dir = os.path.join(os.getcwd(),  "datasets", self.mode)
        if not Path(temp_dir).exists():
            Path(temp_dir).mkdir()

        for idx, entry in enumerate(file_list):
            try:
                volumeNifti, labelNifti = nb.load(entry[0]), nb.load(entry[1])
                volumeDims, labelDims = volumeNifti.shape, labelNifti.shape
            except:
                print('  Removing entry:', entry, 'because files could not be loaded.')
                entry[4] = -1
                continue

            if len(labelDims) == 4:
                print('  4D volume detected: ' + entry[1])
                label = labelNifti.get_fdata()
                label = label.max(axis=3)
                newlabel = nb.Nifti1Image(label, labelNifti.affine, labelNifti.header)
                nb.save(newlabel, entry[1])
                volumeNifti, labelNifti = nb.load(entry[0]), nb.load(entry[1])
                volumeDims, labelDims = volumeNifti.shape, labelNifti.shape

            if not np.array_equal(volumeDims, labelDims):
                entry[4] = -1
                continue

            volume = volumeNifti.get_fdata()
            volume = du.square_and_resize_volume(volume, data_params['target_resolution'])
            volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

            label = np.asarray(labelNifti.dataobj)
            label = np.rint(du.square_and_resize_volume(label, data_params['target_resolution'], True))
            label = np.where(label == 2, 1, label)

            if self.mode != "eval":
                if data_params['remove_black']:
                    volume, label = du.reduce_black_slices_in_volume(volume, label)
                classWeights, weights = preprocessor.estimate_weights_mfb(label)


            entry[0] = str(os.path.join(temp_dir, entry[5] + '_vol.npy'))
            np.save(entry[0], volume)
            entry[1] = str(os.path.join(temp_dir, entry[5] + '_label.npy'))
            np.save(entry[1], label)

            if self.mode != "eval":
                entry[2] = str(os.path.join(temp_dir, entry[2]))
                np.save(entry[2], classWeights)
                entry[3] = str(os.path.join(temp_dir, entry[3]))
                np.save(entry[3], weights)

            pointer += volume.shape[0]
            entry[4] = pointer

            del label, volume, labelNifti, volumeNifti, labelDims, volumeDims

        print('Total number of slices in set: ' + str(pointer))

        self.length = pointer
        self.file_list = [entry for entry in file_list if entry[4] > 0]

    def __getitem__(self, index):
        pointer = 0
        for entry in self.file_list:
            if entry[4] > index:
                idx = index - pointer
                image = np.load(entry[0], mmap_mode='r+')[idx, :, :]
                t_image = torch.unsqueeze(torch.from_numpy(image), 0)
                del image

                label = np.load(entry[1], mmap_mode='r+')[idx, :, :]
                t_label = torch.unsqueeze(torch.from_numpy(label), 0)
                del label

                if self.mode != "eval":
                    class_weight = np.load(entry[2], mmap_mode='r+')[idx, :, :]
                    t_class_weight = torch.unsqueeze(torch.from_numpy(class_weight), 0)
                    del class_weight

                    weight = np.load(entry[3], mmap_mode='r+')
                    t_weight = torch.from_numpy(weight)
                    del weight

                    return t_image, t_label, t_class_weight, t_weight
                else:
                    return t_image, t_label

            else:
                pointer = entry[4]

    def __len__(self):
        return self.length

    def get_volume_numbers(self):
        return len(self.file_list)

    def get_volume_file_name(self, idx):
        volume, label = self.file_list[idx][0], self.file_list[idx][1]
        return volume, label


class MiniDataset(data.Dataset):

    def __init__(self, vol_file, label_file):

        self.vol_file = vol_file
        self.label_file = label_file

        vol_np = np.load(self.vol_file)
        self.length = vol_np.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        image = np.load(self.vol_file, mmap_mode='r+')[idx, :, :]
        t_image = torch.unsqueeze(torch.from_numpy(image), 0)
        del image

        label = np.load(self.label_file, mmap_mode='r+')[idx, :, :]
        t_label = torch.unsqueeze(torch.from_numpy(label), 0)
        del label

        return t_image, t_label


def get_volume_mini_dataloader(vol_file_path, label_file_path, batch_size):

    eval_dataset = MiniDataset(vol_file_path, label_file_path)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=4, pin_memory=True)

    return eval_loader
