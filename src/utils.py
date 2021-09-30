import numpy as np
from pathlib import Path
import os
import shutil
import cv2

from pytz import timezone
from datetime import datetime
from functools import partial


def colorize_depth(img, max_var=2000):
    img_colorized = np.zeros([img.shape[0], img.shape[1], 3]).astype(np.uint8)
    img_colorized[:, :, 1] = 255
    img_colorized[:, :, 2] = 255
    img_hue = img.copy().astype(np.float32)
    img_hue[np.where(img_hue > max_var)] = 0
    zero_idx = np.where((img_hue > max_var) | (img_hue == 0))
    img_hue *= 255.0 / max_var
    img_colorized[:, :, 0] = img_hue.astype(np.uint8)
    img_colorized = cv2.cvtColor(img_colorized, cv2.COLOR_HSV2RGB)
    img_colorized[zero_idx[0], zero_idx[1], :] = 0
    return img_colorized


def get_time():
    utc_now = datetime.now(timezone("UTC"))
    jst_now = utc_now.astimezone(timezone("Asia/Tokyo"))
    time = str(jst_now).split(".")[0].split(" ")[0] + "_" + str(jst_now).split(".")[0].split(" ")[1]
    return time


def make_save_dir(save_dir_path_str):
    if not os.path.exists(save_dir_path_str):
        os.mkdir(save_dir_path_str)


def clean_save_dir(save_dir_path_str):
    if os.path.exists(save_dir_path_str):
        shutil.rmtree(save_dir_path_str)
    os.mkdir(save_dir_path_str)


def count_images(save_dir_path_str):
    return len(list(Path(save_dir_path_str).glob("color_image_*.png")))


def draw_frames(frame, color_image, depth_image, res_image_width, res_image_height):
    depth_image_colorized = colorize_depth(depth_image)
    color_image_viz = cv2.resize(color_image, (res_image_width, res_image_height))
    depth_image_viz = cv2.resize(depth_image_colorized, (res_image_width, res_image_height))
    frame[:res_image_height, :res_image_width, :] = color_image_viz
    frame[:res_image_height, res_image_width : res_image_width * 2, :] = depth_image_viz
    return frame


def _save_images_with_name_and_savepath(image, image_name, save_dir_path_str):
    save_name = str(Path(save_dir_path_str, image_name))
    cv2.imwrite(save_name, image)


def save_images(color_image, depth_image, depth_image_aligned2color, ir_image_left, ir_image_right, save_dir_path_str):
    time = get_time()
    _save_images = partial(_save_images_with_name_and_savepath, save_dir_path_str=save_dir_path_str)
    _save_images(color_image, "color_image_{}.png".format(time))
    _save_images(ir_image_left, "ir_image_left_{}.png".format(time))
    _save_images(ir_image_right, "ir_image_right_{}.png".format(time))
    _save_images(depth_image, "depth_image_{}.png".format(time))
    _save_images(depth_image_aligned2color, "depth_image_aligned2color_{}.png".format(time))