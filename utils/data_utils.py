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

class elastic_deform(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        img = data['image']
        lab = data['label']
        wgt = data['weight']

        x_coo = np.random.randint(100, 300)
        y_coo = np.random.randint(100, 300)
        dx = np.random.randint(10, 40)
        dy = np.random.randint(10, 40)
        if random.random() < self.p:
            img = elastic_deformation(img, x_coo, y_coo, dx, dy)
            lab = elastic_deformation(lab, x_coo, y_coo, dx, dy)
            wgt = elastic_deformation(wgt, x_coo, y_coo, dx, dy)

            lab = np.where(lab <= 20, 0, lab)
            lab = np.where(lab > 20, 255, lab)

        return {'image': img, 'label': lab, 'weight': wgt}


def elastic_deformation(image, x_coord, y_coord, dx, dy):
    """ Applies random elastic deformation to the input image
        with given coordinates and displacement values of deformation points.
        Keeps the edge of the image steady by adding a few frame points that get displacement value zero.
    Input: image: array of shape (N.M,C) (Haven't tried it out for N != M), C number of channels
           x_coord: array of shape (L,) contains the x coordinates for the deformation points
           y_coord: array of shape (L,) contains the y coordinates for the deformation points
           dx: array of shape (L,) contains the displacement values in x direction
           dy: array of shape (L,) contains the displacement values in x direction
    Output: the deformed image (shape (N,M,C))
    """

    image = image.transpose((2, 1, 0))
    ## Preliminaries
    # dimensions of the input image
    shape = image.shape

    # centers of x and y axis
    x_center = shape[1] / 2
    y_center = shape[0] / 2

    ## Construction of the coarse grid
    # deformation points: coordinates

    # anker points: coordinates
    x_coord_anker_points = np.array([0, x_center, shape[1] - 1, 0, shape[1] - 1, 0, x_center, shape[1] - 1])
    y_coord_anker_points = np.array([0, 0, 0, y_center, y_center, shape[0] - 1, shape[0] - 1, shape[0] - 1])
    # anker points: values
    dx_anker_points = np.zeros(8)
    dy_anker_points = np.zeros(8)

    # combine deformation and anker points to coarse grid
    x_coord_coarse = np.append(x_coord, x_coord_anker_points)
    y_coord_coarse = np.append(y_coord, y_coord_anker_points)
    coord_coarse = np.array(list(zip(x_coord_coarse, y_coord_coarse)))

    dx_coarse = np.append(dx, dx_anker_points)
    dy_coarse = np.append(dy, dy_anker_points)

    ## Interpolation onto fine grid
    # coordinates of fine grid
    coord_fine = [[x, y] for x in range(shape[1]) for y in range(shape[0])]
    # interpolate displacement in both x and y direction
    dx_fine = ipol.griddata(coord_coarse, dx_coarse, coord_fine, method='cubic')  # cubic works better but takes longer
    dy_fine = ipol.griddata(coord_coarse, dy_coarse, coord_fine, method='cubic')  # other options: 'linear'
    # get the displacements into shape of the input image (the same values in each channel)


    dx_fine = dx_fine.reshape(shape[0:2])
    dx_fine = np.stack([dx_fine] * shape[2], axis=-1)
    dy_fine = dy_fine.reshape(shape[0:2])
    dy_fine = np.stack([dy_fine] * shape[2], axis=-1)

    ## Deforming the image: apply the displacement grid
    # base grid
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    # add displacement to base grid (-> new coordinates)
    indices = np.reshape(y + dy_fine, (-1, 1)), np.reshape(x + dx_fine, (-1, 1)), np.reshape(z, (-1, 1))
    # evaluate the image at the new coordinates
    #print("input ndim ", image.ndim)
    deformed_image = map_coordinates(image, indices, order=2, mode='nearest')
    deformed_image = deformed_image.reshape(image.shape)

    return deformed_image.transpose((2, 0, 1))

# ====================================================================================================================
# Data Loader Utils
# ====================================================================================================================

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


def load_and_preprocess(file_path, data_load_params):
    volume, labelmap = load_nft_volumes(file_path,  data_load_params)
    volume, labelmap, class_weights, weights = preprocess(volume, labelmap, remap_config=data_load_params['remap_config'],
                                                          reduce_slices=data_load_params['reduce_slices'],
                                                          remove_black=data_load_params['remove_black'],
                                                          return_weights=data_load_params['return_weights'])

    return volume, labelmap, class_weights, weights


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


def preprocess(volume, labelmap, remap_config, reduce_slices=False, remove_black=False, return_weights=False):
    volume = np.clip(volume, -512, 512)
    volume = (volume - np.min(volume)) / np.max((np.max(volume) - np.min(volume), 1e-3))
    labelmap = np.where(labelmap == 2, 1, labelmap)

    if reduce_slices:
        volume, labelmap = preprocessor.reduce_slices(volume, labelmap)

    if remap_config:
        labelmap = preprocessor.remap_labels(labelmap, remap_config)

    if remove_black:
        volume, labelmap = preprocessor.remove_black(volume, labelmap)

    if return_weights:
        class_weights, weights = preprocessor.estimate_weights_mfb(labelmap)
        return volume, labelmap, class_weights, weights
    else:
        return volume, labelmap, None, None


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
