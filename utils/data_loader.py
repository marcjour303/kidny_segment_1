import os

import h5py
import numpy as np
import torch
from torch.utils import data
import utils.kits_data_utils as kutils
import data_utils as du
import pathlib


class SLDataset(data.Dataset):
    def __init__(self, data_params, phase, transforms=None):
        self.transforms = transforms

        for file_path in file_paths:
            volume, labelmap, class_weights, weights, header = du.load_and_preprocess(file_path, orientation,
                                                                                   remap_config=remap_config,
                                                                                   reduce_slices=reduce_slices,
                                                                                   remove_black=remove_black,
                                                                                   return_weights=return_weights,
                                                                                   downsample=downsample)

            volume_list.append(volume)
            labelmap_list.append(labelmap)

            if return_weights:
                class_weights_list.append(class_weights)
                weights_list.append(weights)

            headers.append(header)

            print("#", end='', flush=True)
        print("100%", flush=True)

        if phase == 'train':
            data_files = h5py.File(os.path.join(data_params['data_dir'], data_params['train_data_file']), 'r')
            labels = h5py.File(os.path.join(data_params['data_dir'], data_params['train_label_file']), 'r')
            class_weights = h5py.File(os.path.join(data_params['data_dir'], data_params['train_class_weights_file']), 'r')
            weights = h5py.File(os.path.join(data_params['data_dir'], data_params['train_weights_file']), 'r')

        if phase == 'val':
            data_files = h5py.File(os.path.join(data_params['data_dir'], data_params['test_data_file']), 'r')
            labels = h5py.File(os.path.join(data_params['data_dir'], data_params['test_label_file']), 'r')
            class_weights = h5py.File(os.path.join(data_params['data_dir'], data_params['test_class_weights_file']), 'r')
            weights = h5py.File(os.path.join(data_params['data_dir'], data_params['test_weights_file']), 'r')

        self.data_files = data_files
        self.labels = labels
        self.class_weights = class_weights
        self.weights = weights

    def __getitem__(self, index):

        self.X = self.data_files['data'][index]
        self.y = self.labels['label'][index]
        self.w = self.class_weights['class_weights'][index]

        img = self.X
        label = self.y
        weight = self.w

        label_3d = np.expand_dims(label, axis=0)
        weight_3d = np.expand_dims(weight, axis=0)

        sample = {'image': img, 'label': label_3d, 'weight': weight_3d}

        if self.transforms is not None:
            sample = self.transforms(sample)

        img = sample['image'].unsqueeze(dim=0)

        label = sample['label'].squeeze()
        weight = sample['weight'].squeeze()

        return img, label, weight

    def __len__(self):
        return len(self.labels['label'])

def apply_split(data_split, data_dir):
    file_paths = kutils.load_file_paths(data_dir)
    print("Total no of volumes to process : %d" % len(file_paths))
    train_ratio, test_ratio = data_split.split(",")
    train_len = int((int(train_ratio) / 100) * len(file_paths))
    train_idx = np.random.choice(len(file_paths), train_len, replace=False)
    val_idx = np.array([i for i in range(len(file_paths)) if i not in train_idx])
    train_file_paths = [file_paths[i] for i in train_idx]
    val_file_paths = [file_paths[i] for i in val_idx]

    #train_data['cases'] = np.array(train_file_paths, dtype=int).tolist()

    train_data = {}
    train_data['cases'] = train_file_paths
    kutils.write_to_file("..\\datasets\\train_data.json", train_data)

    val_data = {}
    val_data['cases'] = val_file_paths
    kutils.write_to_file("..\\datasets\\val_data.json", val_data)

    return train_file_paths, val_file_paths

class Dataloder_img(data.Dataset):
    def __init__(self, data_dir, data_params, phase, transforms, ):

        self.transforms = transforms
        self.train_files, self.val_files = apply_split(data_params["data_split"], data_dir)
        self.phase = phase

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        label_name = self.lables[idx]
        if self.phase=='train':
            volume, labelmap, class_weights, weights, header = load_and_preprocess(self.train_files[idx], orientation,
                                                                                   remap_config=remap_config,
                                                                                   reduce_slices=reduce_slices,
                                                                                   remove_black=remove_black,
                                                                                   return_weights=return_weights,
                                                                                   downsample=downsample)
        img = nib.load(os.path.join(self.root_dir, img_name))  # !Image.open(os.path.join(self.root_dir,img_name))
        # change to numpy
        img = np.array(img.dataobj)
        # change to PIL
        img = Image.fromarray(img.astype('uint8'), 'RGB')

        print(img.size)

        label = nib.load(os.path.join(self.seg_dir, label_name))  # !Image.open(os.path.join(self.seg_dir,label_name))
        # change to numpy
        label = np.array(label.dataobj)
        # change to PIL
        label = Image.fromarray(label.astype('uint8'), 'RGB')

        print(label.size)

        if self.transforms:
            img = self.transforms(img)
            label = self.transforms(label)
            return img, label
        else:
            return img, label


full_dataset = Dataloder_img(' image ',
                             ' labels ', tfms.Compose([tfms.RandomRotation(180), tfms.ToTensor()
                                                       ]))  #

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
train_loader = data.DataLoader(train_dataset, shuffle=False, batch_size=bs)
val_loader = data.DataLoader(val_dataset, shuffle=False, batch_size=bs)

test_img, test_lb = next(iter(full_dataset))
print(test_img[0].shape)
plt.imshow(test_img[0])
plt.show()
