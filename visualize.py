from pathlib import Path
import numpy as np
from imageio import imwrite

import cv2
import os
import utils.viz_utils as vu



def visualize(case_name, vol, seg, destination, hu_min=vu.DEFAULT_HU_MIN, hu_max=vu.DEFAULT_HU_MAX,
              k_color=vu.DEFAULT_KIDNEY_COLOR, t_color=vu.DEFAULT_TUMOR_COLOR,
              alpha=vu.DEFAULT_OVERLAY_ALPHA):

    # Prepare output location
    out_path = Path(destination)
    if not out_path.exists():
        out_path.mkdir()

    print("Volume with id: ", case_name)
    # Convert to a visual format
    vol_ims = vu.hu_to_grayscale(vol, hu_min, hu_max)
    seg_ims = vu.class_to_color(seg, k_color, t_color)

    print("creating video...")

    # Overlay the segmentation colors
    viz_ims = vu.overlay(vol_ims, seg_ims, seg, alpha)
    height, width, layers = viz_ims[0].shape

    frame_rate = 10

    video_file_name = (str(out_path / case_name) + "_pred.mp4v")
    print(video_file_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_file_name, fourcc, frame_rate, (width, height))
    print(vol.shape)
    print(viz_ims.shape)
    for i in range(viz_ims.shape[0]):
        video.write(viz_ims[i])
    video.release()

    if os.path.exists(video_file_name):
        print("video file created")
    else:
        print("The video file failed to get created")
        png_path = out_path / case_name
        png_path.mkdir()
        for i in range(viz_ims.shape[0]):
            fpath = png_path / ("{:05d}.png".format(i))
            imwrite(str(fpath), viz_ims[i])
