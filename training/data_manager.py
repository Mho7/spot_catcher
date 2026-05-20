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


def scan_and_register():
    normal = [f for f in glob.glob(os.path.join(STATIC_DIR, f"{NORMAL_PREFIX}*.png"))
              if "_ov_" not in os.path.basename(f)]
    defect = [f for f in glob.glob(os.path.join(STATIC_DIR, f"{DEFECT_PREFIX}*.png"))
              if "_ov_" not in os.path.basename(f)]
    register_images([(f, "normal") for f in normal] + [(f, "defect") for f in defect])
    return len(normal), len(defect)


def prepare_split(val_ratio=0.2, seed=42):
    unused = get_unused_images()
    normal = unused["normal"]
    rng = random.Random(seed)
    rng.shuffle(normal)
    split = max(1, int(len(normal) * (1 - val_ratio)))
    return {
        "train_normal": normal[:split],
        "val_normal":   normal[split:],
        "val_defect":   unused["defect"],
    }
