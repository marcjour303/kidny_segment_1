import os

from pathlib import Path
import nibabel as nb
import numpy as np

import utils.preprocessor as preprocessor
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy import interpolate as ipol
from sklearn.utils import shuffle
import utils.kits_data_utils as kutils


from skimage.transform import resize


def square_and_resize_volume(volume, targetResolution, nearestNeighbor=False, debug=False):
    if debug:
        print('    Resizing from ' + str(volume.shape[0]) + 'x' + str(volume.shape[1]) + 'x' + str(volume.shape[2]) +
              ' to ' + str(volume.shape[0]) + 'x' + str(targetResolution) + 'x' + str(targetResolution))

    if nearestNeighbor:
        volume = resize(volume, (volume.shape[0], targetResolution, targetResolution), mode='constant', cval=0,
                        clip=True, preserve_range=True, anti_aliasing=False, order=0)
    else:
        volume = resize(volume, (volume.shape[0], targetResolution, targetResolution), mode='constant', cval=0,
                        clip=True, preserve_range=True, anti_aliasing=False)
    if debug:
        print("Done resizing")
    return volume


def reduce_black_slices_in_volume(volume, label, threshold=10):
    slicesToDelete = []
    for i in range(label.shape[0]):
        slice = label[i, :, :]
        if slice.max() == 0:
            remove = True
            for j in range(max([0, i - threshold]), min([i + threshold, label.shape[0]])):
                neighboringSlice = label[j, :, :]
                if neighboringSlice.max() == 1:
                    remove = False
                    break
            if remove:
                slicesToDelete.append(i)

    return np.delete(volume, slicesToDelete, axis=0), np.delete(label, slicesToDelete, axis=0)


def load_nft_volumes(file_path, load_params):
    print("Loading vol: %s with label: %s and resolution %d" % (file_path[0], file_path[1], load_params['target_resolution']))

    volume = np.squeeze(nb.load(file_path[0]).get_fdata())
    labelmap = np.squeeze(nb.load(file_path[1]).get_fdata())

    print("Volume shape: ", volume.shape)
    print("Lable shape: ", labelmap.shape)

    volume, labelmap = preprocessor.rotate_orientation(volume, labelmap, load_params['orientation'])

    volume = square_and_resize_volume(volume, load_params['target_resolution'], nearestNeighbor=False)
    labelmap = square_and_resize_volume(labelmap, load_params['target_resolution'], nearestNeighbor=False)

    # shuffle volume and label slices
    volume, labelmap = shuffle(volume, labelmap)

    return volume, labelmap


def load_volume_paths_from_case_file(data_dir, file_path):
    data_dir = Path(data_dir)
    volumes_to_use = kutils.get_case_numbers_from_file(file_path)

    vol_files = [
        [os.path.join(kutils.get_case_path(data_dir, case), 'imaging.nii.gz'),
         os.path.join(kutils.get_case_path(data_dir, case), 'segmentation.nii.gz')]
        for
        case in volumes_to_use]
    return vol_files


def filter_and_split_data(data_params):
    data_skip_file , test_data_file = data_params["data_skip"], data_params["test_data"]
    train_file, val_file = data_params["train_data_file"], data_params["val_data_file"],
    data_split, data_dir = data_params["data_split"], data_params["data_dir"]

    data_skip = kutils.get_case_numbers_from_file(data_skip_file)
    test_data = kutils.get_case_numbers_from_file(test_data_file)
    case_numbers = kutils.filter_case_numbers(data_skip, test_data, data_dir)

    print("Total no of volumes to process : %d" % len(case_numbers))
    train_ratio, test_ratio = data_split.split(",")
    train_len = int((int(train_ratio) / 100) * len(case_numbers))
    train_idx = np.random.choice(len(case_numbers), train_len, replace=False)
    val_idx = np.array([i for i in range(len(case_numbers)) if i not in train_idx])
    train_cases = [case_numbers[i] for i in train_idx]
    val_cases = [case_numbers[i] for i in val_idx]

    train_data = {}
    train_data['cases'] = train_cases
    kutils.write_to_file(train_file, train_data)


    val_data = {}
    val_data['cases'] = val_cases
    kutils.write_to_file(val_file, val_data)

    return
