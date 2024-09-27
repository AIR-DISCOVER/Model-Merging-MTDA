import os
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import multiprocessing
import shutil

source_dir = "/DATA_EDS/gaoha/datasets/IDD/"
target_dir = "/home/chenht/datasets/cityscapes_idd_combined/"

gt_lists_train = glob.glob(f"{source_dir}*/gtFine/train/*/*labelcsTrainIds*")
target_dir_gt_train = f"{target_dir}gtFine/train/"
rgb_lists_train = glob.glob(f"{source_dir}*/leftImg8bit/train/*/*leftImg8bit*")
target_dir_rgb_train = f"{target_dir}leftImg8bit/train/"

gt_lists_val = glob.glob(f"{source_dir}*/gtFine/val/*/*labelcsTrainIds*")
target_dir_gt_val = f"{target_dir}gtFine/val/"
rgb_lists_val = glob.glob(f"{source_dir}*/leftImg8bit/val/*/*leftImg8bit*")
target_dir_rgb_val = f"{target_dir}leftImg8bit/val/"


def process_image(args):
    gt_file, rgb_file, target_dir_gt, target_dir_rgb = args
    new_gt = f"{gt_file.split('/')[5]}_{gt_file.split('/')[8]}_{'_'.join(gt_file.split('/')[9].split('_')[:-1])}"
    if gt_file.endswith("png"):
        shutil.copy(gt_file, os.path.join(target_dir_gt, f"{new_gt}_labelTrainIds.png"))
    else:
        raise NotImplementedError()  # IDD gtFine does not contain non-png files
    
    new_rgb = f"{rgb_file.split('/')[5]}_{rgb_file.split('/')[8]}_{'_'.join(rgb_file.split('/')[9].split('_')[:-1])}"
    if rgb_file.endswith("png"):
        shutil.copy(rgb_file, os.path.join(target_dir_rgb, f"{new_rgb}_leftImg8bit.png"))
    elif rgb_file.endswith("jpg"):
        image = Image.open(rgb_file)
        image.save(os.path.join(target_dir_rgb, f"{new_rgb}_leftImg8bit.png"), "PNG")
    else:
        raise NotImplementedError()

assert len(gt_lists_train) == len(rgb_lists_train)
assert len(gt_lists_val) == len(rgb_lists_val)

with multiprocessing.Pool() as pool:
    results_train = list(tqdm(pool.imap(process_image, zip(gt_lists_train, rgb_lists_train, [target_dir_gt_train]*len(gt_lists_train), [target_dir_rgb_train]*len(rgb_lists_train))), total=len(gt_lists_train)))
    results_val = list(tqdm(pool.imap(process_image, zip(gt_lists_val, rgb_lists_val, [target_dir_gt_val]*len(gt_lists_val), [target_dir_rgb_val]*len(rgb_lists_val))), total=len(gt_lists_val)))
