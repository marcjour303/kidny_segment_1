import os

import h5py
import numpy as np
import torch
from torch.utils import data
import utils.preprocessor as preprocessor
import utils.data_utils as du

import matplotlib.pyplot as plt
from pathlib import Path


from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import nibabel as nb
from PIL import Image
import random


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


class DataloaderNII(data.Dataset):
    def __init__(self, file_list, data_params, phase, transforms):

        self.transforms = transforms
        self.phase = phase
        # convert files to h5
        if phase == 'train':
            dest_folder = os.path.join(os.getcwd(), "datasets", "h5_train")
        elif phase == 'val':
            dest_folder = os.path.join(os.getcwd(), "datasets", "h5_val")


        self.data_files = h5py.File(os.path.join(dest_folder, "Data.h5"), 'r')
        self.labels = h5py.File(os.path.join(dest_folder, "Label.h5"), 'r')
        self.class_weights = h5py.File(os.path.join(dest_folder, "Class_Weight.h5"), 'r')
        self.weights = h5py.File(os.path.join(dest_folder, "Weight.h5"), 'r')

    def __len__(self):
        return len(self.labels['label'])

    def __getitem__(self, index):

        self.X = self.data_files['data'][index]
        self.y = self.labels['label'][index]
        self.class_w = self.class_weights['class_weights'][index]

        #weight_index = self.get_weight_index(index)
        #self.w = self.weights['weights'][weight_index]

        img = self.X
        label = self.y
        class_weight = self.class_w

        print("Input shape: ",self.X.shape)
        print("Y shape: ", self.y.shape)
        print("weight shape: ", self.class_w.shape)

        #weights = np.array(self.w, dtype=np.float64)

        #out_path = Path(os.path.join("E:\\", "label_vis"))
        #if not out_path.exists():
        #    out_path.mkdir()

        #if np.count_nonzero(label == 1) > 10:
        #    val = random.randint(0, 200)
        #    fig, ax = plt.subplots(1, 2)
        #    _ = ax[0].imshow(label, cmap='Greys', vmax=abs(label).max(), vmin=abs(label).min())
        #    _ = ax[1].imshow(img, cmap='Greys', vmax=abs(img).max(), vmin=abs(img).min())
        #    fig_path = os.path.join(out_path, self.phase + "_img_" + str(val) + '.jpeg')
        #    print(str(fig_path))
        #    fig.savefig(str(fig_path))

        label_3d = np.expand_dims(label, axis=0)
        class_weight_3d = np.expand_dims(class_weight, axis=0)

        sample = {'image': img, 'label': label_3d, 'class_weight': class_weight_3d}

        if self.transforms is not None:
            sample = self.transforms(sample)

        img = sample['image'].unsqueeze(dim=0)

        label = sample['label'].squeeze()
        class_weight = sample['class_weight'].squeeze()

        return img, label, class_weight

    def get_weight_index(self, index):
        for i in range(len(self.index_weight_idx)):
            if self.index_weight_idx[i] > index:
                return i

    def close_files(self):
        self.data_files.close()
        self.labels.close()
        self.class_weights.close()
        self.weights.close()


class NiftiData(data.Dataset):

    def __init__(self, filePaths, data_params, train=True):

        path_to_files = os.path.split(filePaths[0][0])[0]
        case_id = os.path.split(path_to_files)[1]
        file_list = [[vol[0],
                      vol[1],
                      case_id + '_class_weights.npy',
                      case_id + '_weights.npy',
                      0] for vol in filePaths]

        pointer = 0
        dir_path = None
        if train:
            dir_path = "h5_train"
        else:
            dir_path = "h5_val"
        temp_dir = os.path.join(os.getcwd(),  "datasets", dir_path)

        if not Path(temp_dir).exists():
            Path(temp_dir).mkdir()

        for entry in file_list:
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
                print('    Successfully converted to 3D')

            if not np.array_equal(volumeDims, labelDims):
                entry[4] = -1
                print('  Dimension mismatch,', volumeDims, labelDims, 'removing entry:', entry)
                continue

            print('  Processing volume file: ' + entry[0])
            volume = volumeNifti.get_fdata()
            print("Data shape: ", volume.shape)
            volume = du.square_and_resize_volume(volume, data_params['target_resolution'])
            volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

            print('    Processing label file: ' + entry[1])
            label = np.asarray(labelNifti.dataobj)
            label = np.rint(du.square_and_resize_volume(label, data_params['target_resolution'], True))

            if data_params['remove_black']:
                volume, label = du.reduce_black_slices_in_volume(volume, label)
                print('    Slices reduced from ' + str(volumeDims) + ' to ' + str(volume.shape))

            print('    Generating weights')
            classWeights, weights = preprocessor.estimate_weights_mfb(label)

            print('    Saving as numpy files')
            entry[0] = str(os.path.join(temp_dir, case_id + '_vol.npy'))
            np.save(entry[0], volume)
            entry[1] = str(os.path.join(temp_dir, case_id + '_label.npy'))
            label = np.where(label == 2, 1, label)
            np.save(entry[1], label)
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
                image = np.load(entry[0], mmap_mode='r+')[index - pointer, :, :]
                t_image = torch.unsqueeze(torch.from_numpy(image.copy()), 0)
                del image

                label = np.load(entry[1], mmap_mode='r+')[index - pointer, :, :]
                t_label = torch.from_numpy(label.copy())
                del label

                classWeight = np.load(entry[2], mmap_mode='r+')[index - pointer, :, :]
                t_classWeight = torch.from_numpy(classWeight.copy())
                del classWeight

                weight = np.load(entry[3], mmap_mode='r+')
                t_weight = torch.from_numpy(weight.copy())
                del weight

                return t_image, t_label, t_classWeight, t_weight

            else:
                pointer = entry[4]

    def __len__(self):
        return self.length
