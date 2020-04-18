import os

import numpy as np
import nibabel as nib
from pathlib import Path

def get_full_case_id(cid):
    try:
        cid = int(cid)
        case_id = "case_{:05d}".format(cid)
    except ValueError:
        case_id = cid

    return case_id


def get_case_path(cid):
    # Resolve location where data should be living
    data_path = Path("E:\kits19\data")

    if not data_path.exists():
        raise IOError(
            "Data path, {}, could not be resolved".format(str(data_path))
        )

    # Get case_id from provided cid
    case_id = get_full_case_id(cid)

    # Make sure that case_id exists under the data_path
    case_path = data_path / case_id
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
    print(str(case_path / "only_kidney_seg.nii.gz"))
    img = nib.Nifti1Image(seg, np.eye(4))
    nib.save(img, str(case_path / "only_kidney_seg.nii.gz"))
    return
