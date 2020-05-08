import os

import numpy as np
import nibabel as nib
from pathlib import Path
from imageio import imwrite

# Constants
DEFAULT_KIDNEY_COLOR = [255, 0, 0]
DEFAULT_TUMOR_COLOR = [0, 0, 255]
DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512
DEFAULT_OVERLAY_ALPHA = 0.3
DEFAULT_PLANE = "axial"

def get_full_case_id(cid):
    try:
        cid = int(cid)
        case_id = "case_{:05d}".format(cid)
    except ValueError:
        case_id = cid

    return case_id


def get_case_path(cid):
    # Resolve location where data should be living
    data_path = Path("E:\\kits19\\data")

    if not data_path.exists():
        raise IOError(
            "Data path, {}, could not be resolved".format(str(data_path))
        )

    # Get case_id from provided cid
    case_id = get_full_case_id(cid)

    # Make sure that case_id exists under the data_path
    case_path = os.path.join(data_path, case_id)
    print("data path: ", str(data_path))
    print("case id: ", case_id)
    if not case_path.exists():
        raise ValueError(
            "Case could not be found \"{}\"".format(case_path.name)
        )

    return case_path


def load_volume(cid):
    case_path = get_case_path(cid)
    vol = nib.load(str(case_path / "imaging.nii.gz"))
    return vol


def load_segmentation(cid):
    case_path = get_case_path(cid)
    seg = nib.load(str(case_path / "only_kidney_seg.nii.gz"))
    print(str(case_path / "only_kidney_seg.nii.gz"))

    return seg


def load_prediction(path, cid):
    case_id = get_full_case_id(cid)
    case_path = os.path.join(path, case_id + str('_pred.nii'))
    seg = nib.load(case_path)
    print(case_path)
    return seg


def load_case(cid):
    vol = load_volume(cid)
    seg = load_segmentation(cid)
    return vol, seg


def save_only_kidney_seg(seg, cid):
    case_path = get_case_path(cid)
    img = nib.Nifti1Image(seg, np.eye(4))
    nib.save(img, str(case_path / "only_kidney_seg.nii.gz"))
    return


def hu_to_grayscale(volume, hu_min, hu_max):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval) / max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255 * im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)


def class_to_color(segmentation, k_color, t_color):
    # initialize output to zeros
    shp = segmentation.shape
    seg_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)

    # set output to appropriate color at each location

    seg_color[np.equal(segmentation, 1)] = k_color
    seg_color[np.equal(segmentation, 2)] = t_color
    return seg_color


def overlay(volume_ims, segmentation_ims, segmentation, alpha):
    # Get binary array for places where an ROI lives
    segbin = np.greater(segmentation, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay
    overlayed = np.where(
        repeated_segbin,
        np.round(alpha * segmentation_ims + (1 - alpha) * volume_ims).astype(np.uint8),
        np.round(volume_ims).astype(np.uint8)
    )

    return overlayed


def visualize_slice(destination, name, vol, label):
    # Prepare output location
    out_path = Path(destination)
    if not out_path.exists():
        out_path.mkdir()


    vol_ims = hu_to_grayscale(vol, DEFAULT_HU_MIN, DEFAULT_HU_MAX)
    seg_ims = class_to_color(label, DEFAULT_KIDNEY_COLOR, DEFAULT_TUMOR_COLOR)

    viz_ims = overlay(vol_ims, seg_ims, label, DEFAULT_OVERLAY_ALPHA)

    kidney_cell_count = np.count_nonzero(np.equal(label, 1))

    viz_ims = viz_ims.squeeze()
    print(viz_ims.shape)

    fpath = out_path / ("{:05d}.png".format(name))
    imwrite(str(fpath), viz_ims)