"""static/ 스캔 및 train/val 분리."""
import os
import sys
import glob
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from config import STATIC_DIR
from db_training import register_images, get_unused_images

NORMAL_PREFIX = "normal_"
DEFECT_PREFIX = "cam_"

GLASS_FORMAT_DIR = os.path.join(os.path.dirname(__file__), "..", "glass_format", "spot")


def scan_and_register():
    normal = [f for f in glob.glob(os.path.join(STATIC_DIR, f"{NORMAL_PREFIX}*.png"))
              if "_ov_" not in os.path.basename(f)]
    defect = [f for f in glob.glob(os.path.join(STATIC_DIR, f"{DEFECT_PREFIX}*.png"))
              if "_ov_" not in os.path.basename(f)]
    register_images([(f, "normal") for f in normal] + [(f, "defect") for f in defect])
    return len(normal), len(defect)


def get_defect_with_masks():
    """glass_format/spot/test/bad/ 에서 결함 이미지와 대응 마스크 경로를 반환."""
    bad_dir = os.path.join(GLASS_FORMAT_DIR, "test", "bad")
    gt_dir = os.path.join(GLASS_FORMAT_DIR, "ground_truth", "bad")
    if not os.path.isdir(bad_dir) or not os.path.isdir(gt_dir):
        return [], []
    defect_paths, mask_paths = [], []
    for fname in sorted(os.listdir(bad_dir)):
        img_path = os.path.join(bad_dir, fname)
        mask_path = os.path.join(gt_dir, fname)
        if os.path.isfile(mask_path):
            defect_paths.append(img_path)
            mask_paths.append(mask_path)
    return defect_paths, mask_paths


def prepare_split(val_ratio=0.2, seed=42):
    unused = get_unused_images()
    normal = unused["normal"]
    rng = random.Random(seed)
    rng.shuffle(normal)
    split = max(1, int(len(normal) * (1 - val_ratio)))
    train_defect, train_defect_mask = get_defect_with_masks()
    return {
        "train_normal": normal[:split],
        "val_normal":   normal[split:],
        "val_defect":   unused["defect"],
        "train_defect": train_defect,
        "train_defect_mask": train_defect_mask,
    }
