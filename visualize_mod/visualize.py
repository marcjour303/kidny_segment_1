from pathlib import Path
import argparse

import scipy.misc
import numpy as np
import imageio
import json
from imageio import imwrite

import visualize_mod.utils as vu
import cv2
import glob
import os
from settings import Settings as stng

# Constants
DEFAULT_KIDNEY_COLOR = [255, 0, 0]
DEFAULT_TUMOR_COLOR = [0, 0, 255]
DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512
DEFAULT_OVERLAY_ALPHA = 0.3
DEFAULT_PLANE = "axial"


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


def visualize(vid, pred_path, destination, hu_min=DEFAULT_HU_MIN, hu_max=DEFAULT_HU_MAX,
              k_color=DEFAULT_KIDNEY_COLOR, t_color=DEFAULT_TUMOR_COLOR,
              alpha=DEFAULT_OVERLAY_ALPHA, plane=DEFAULT_PLANE):
    plane = plane.lower()

    plane_opts = ["axial", "coronal", "sagittal"]
    if plane not in plane_opts:
        raise ValueError((
                             "Plane \"{}\" not understood. "
                             "Must be one of the following\n\n\t{}\n"
                         ).format(plane, plane_opts))

    # Prepare output location
    out_path = Path(destination)
    if not out_path.exists():
        out_path.mkdir()

    # Load segmentation and volume
    vol = vu.load_volume(vid)
    seg = vu.load_prediction(pred_path, vid)
    spacing = vol.affine
    vol = vol.get_fdata()
    seg = seg.get_fdata()



    print("Estimate background vals: ", np.count_nonzero(seg == 0))
    print("Estimate kidney vals: ", np.count_nonzero(seg == 1))
    print("Estimate tumor vals: ", np.count_nonzero(seg == 2))

    vol = np.squeeze(vol[::4, ::4, ::4])

    gt_seg = vu.load_segmentation(cid)
    gt_seg = gt_seg.get_fdata()

    print("GT background vals: ", np.count_nonzero(gt_seg == 0))
    print("GT kidney vals: ", np.count_nonzero(gt_seg == 1))
    print("GT tumor vals: ", np.count_nonzero(gt_seg == 2))

    gt_seg = np.squeeze(gt_seg[::4, ::4, ::4])
    gt_seg = gt_seg.astype(np.int32)
    print("check shape matches:")
    print(gt_seg.shape)
    print(seg.shape)
    seg = seg.astype(np.int32)

    # Convert to a visual format
    vol_ims = hu_to_grayscale(vol, hu_min, hu_max)
    seg_ims = class_to_color(seg, k_color, t_color)

    print("creating video...")

    # Save individual images to disk
    if plane == plane_opts[0]:
        # Overlay the segmentation colors
        viz_ims = overlay(vol_ims, seg_ims, seg, alpha)

        height, width, layers = viz_ims[0].shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_rate = 50
        # Get case_id from provided cid
        case_id = vu.get_full_case_id(vid)
        video_file_name = (str(out_path / case_id) + "_pred.mp4v")
        print(video_file_name)
        video = cv2.VideoWriter(video_file_name, fourcc, frame_rate, (width, height))
        list_files = range(viz_ims.shape[0])
        for i in list_files:
            video.write(viz_ims[i])
        video.release()

    if plane == plane_opts[1]:
        # I use sum here to account for both legacy (incorrect) and 
        # fixed affine matrices
        spc_ratio = np.abs(np.sum(spacing[2, :])) / np.abs(np.sum(spacing[0, :]))
        for i in range(vol_ims.shape[1]):
            fpath = out_path / ("{:05d}.png".format(i))
            vol_im = scipy.misc.imresize(
                vol_ims[:, i, :], (
                    int(vol_ims.shape[0] * spc_ratio),
                    int(vol_ims.shape[2])
                ), interp="bicubic"
            )
            seg_im = scipy.misc.imresize(
                seg_ims[:, i, :], (
                    int(vol_ims.shape[0] * spc_ratio),
                    int(vol_ims.shape[2])
                ), interp="nearest"
            )
            sim = scipy.misc.imresize(
                seg[:, i, :], (
                    int(vol_ims.shape[0] * spc_ratio),
                    int(vol_ims.shape[2])
                ), interp="nearest"
            )
            viz_im = overlay(vol_im, seg_im, sim, alpha)
            imwrite(str(fpath), viz_im)

    if plane == plane_opts[2]:
        # I use sum here to account for both legacy (incorrect) and 
        # fixed affine matrices
        spc_ratio = np.abs(np.sum(spacing[2, :])) / np.abs(np.sum(spacing[1, :]))
        for i in range(vol_ims.shape[2]):
            fpath = out_path / ("{:05d}.png".format(i))
            vol_im = scipy.misc.imresize(
                vol_ims[:, :, i], (
                    int(vol_ims.shape[0] * spc_ratio),
                    int(vol_ims.shape[1])
                ), interp="bicubic"
            )
            seg_im = scipy.misc.imresize(
                seg_ims[:, :, i], (
                    int(vol_ims.shape[0] * spc_ratio),
                    int(vol_ims.shape[1])
                ), interp="nearest"
            )
            sim = scipy.misc.imresize(
                seg[:, :, i], (
                    int(vol_ims.shape[0] * spc_ratio),
                    int(vol_ims.shape[1])
                ), interp="nearest"
            )
            viz_im = overlay(vol_im, seg_im, sim, alpha)
            imwrite(str(fpath), viz_im)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-c", "--case_id", required=False, default="None",
        help="The identifier for the case you would like to visualize"
    )
    parser.add_argument(
        "-m", "--mode", required=True, default="test",
        help="The kind of data we want to visualize"
    )
    parser.add_argument(
        "-u", "--upper_hu_bound", required=False, default=DEFAULT_HU_MAX,
        help="The upper bound at which to clip HU values"
    )
    parser.add_argument(
        "-l", "--lower_hu_bound", required=False, default=DEFAULT_HU_MIN,
        help="The lower bound at which to clip HU values"
    )
    parser.add_argument(
        "-p", "--plane", required=False, default=DEFAULT_PLANE,
        help=(
            "The plane in which to visualize the data"
            " (axial, coronal, or sagittal)"
        )
    )
    args = parser.parse_args()

    if args.cid is None:
        settings_file = 'E:\quicknat-master\visualize_mod\vis_settings.ini'
        settings_dictionary = stng(settings_file).settings_dict
        train_params, test_params, val_params = settings_dictionary['TRAIN'], settings_dictionary['TEST'], settings_dictionary['VALIDATE']

        if args.mode == "Train":
            c_params = train_params
        if args.mode == "Validate":
            c_params = val_params
        if args.mode == "Test":
            c_params = test_params

    dat_dir = c_params["data_dir"]
    data_dest = c_params["vis_loc"]

    try:
        with open(eval_file, 'r') as tst:
            test_case_ids = json.load(tst)
            volumes_to_use = test_case_ids['test_cases']
    except IOError:
        print("No evaluation data")

    prediction_files = glob.glob(pathToPred + '*.nii')

    # Run visualization
    destination = os.path.join(pathToPred, 'visu')
    print(destination)

    for cid in volumes_to_use:
        print("Processing: ", cid)
        visualize(cid, data_dir, data_dest,
                  hu_min=args.lower_hu_bound, hu_max=args.upper_hu_bound,
                  plane=args.plane
                  )
    print("Done")
