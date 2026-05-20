"""
외부 GLASS 프로젝트의 모델 코드와 체크포인트로 추론만 실행합니다.

사용법:
    python infer_glass.py --image data/test/bad/frame_0050.png --no-sam
    python infer_glass.py --checkpoint /Users/parkhyunsik/파이썬/GLASS/ckpt_best_173.pth --image sample.png
    python infer_glass.py
"""
import argparse
import glob
import importlib
import os
import re
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

<<<<<<< HEAD
from config import (
    GLASS_BACKBONE,
    GLASS_DSC_HIDDEN,
    GLASS_DSC_LAYERS,
    GLASS_INPUT_SIZE,
    GLASS_LAYERS,
    GLASS_PATCHSIZE,
    GLASS_PRE_PROJ,
    GLASS_PRETRAIN_EMBED_DIM,
    GLASS_PROJECT_DIR,
    GLASS_RESULTS_DIR,
    GLASS_TARGET_EMBED_DIM,
    GLASS_TEST_BAD_DIR,
    GLASS_TEST_GOOD_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SAVE_DIR,
)
=======
# spot_catcher의 utils를 먼저 import (GLASS의 utils와 충돌 방지)
from utils.visual import save_result_image
from utils.preprocessing import SAMMasker

# GLASS 프로젝트 모듈 import
GLASS_DIR = r"C:\Users\User\github\GLASS"
sys.path.insert(0, GLASS_DIR)

import backbones
from glass import GLASS

# run-spot.sh 기준 설정
CKPT_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "ckpt_best_37.pth")
TEST_BAD_DIR = os.path.join(os.path.dirname(__file__), "data", "glass_format", "spot", "test", "bad", "테스트")
TEST_GOOD_DIR = os.path.join(os.path.dirname(__file__), "data", "glass_format", "spot", "test", "good", "real")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "static", "glass_results")

# GLASS run-spot.sh 파라미터
IMAGESIZE_H = 1080
IMAGESIZE_W = 1920
BACKBONE_NAME = "wideresnet50"
LAYERS = ["layer2", "layer3"]
PRETRAIN_EMBED_DIM = 1536
TARGET_EMBED_DIM = 1536
PATCHSIZE = 3
DSC_LAYERS = 2
DSC_HIDDEN = 1024
PRE_PROJ = 1

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
>>>>>>> c274989160b7f55de382aec5a6c7e49114cbd6f1


class CLAHE:
    """CLAHE preprocessing transform for PIL images."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)


def load_glass_modules(glass_dir):
    if not os.path.isdir(glass_dir):
        raise FileNotFoundError(f"GLASS 프로젝트 폴더가 없습니다: {glass_dir}")

    module_names = ("backbones", "common", "glass", "loss", "metrics", "model", "utils")
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    saved_path = list(sys.path)

    try:
        for name in module_names:
            sys.modules.pop(name, None)
        sys.path.insert(0, glass_dir)
        backbones = importlib.import_module("backbones")
        glass_module = importlib.import_module("glass")
        return backbones, glass_module.GLASS
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
            if saved_modules[name] is not None:
                sys.modules[name] = saved_modules[name]
        sys.path[:] = saved_path


def checkpoint_sort_key(path):
    match = re.search(r"ckpt_best_(\d+)\.pth$", os.path.basename(path))
    epoch = int(match.group(1)) if match else -1
    return epoch, os.path.getmtime(path)


def find_checkpoint(path=None, glass_dir=GLASS_PROJECT_DIR):
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"체크포인트가 없습니다: {path}")
        return path

    candidates = []
    candidates.extend(glob.glob(os.path.join(glass_dir, "**", "ckpt_best_*.pth"), recursive=True))
    candidates.extend(glob.glob(os.path.join(SAVE_DIR, "ckpt_best_*.pth")))

    if not candidates:
        raise FileNotFoundError(
            "GLASS 체크포인트를 찾지 못했습니다. "
            f"{glass_dir} 아래 또는 {SAVE_DIR}에 ckpt_best_*.pth를 넣어주세요."
        )
    return max(candidates, key=checkpoint_sort_key)


def build_transform():
    height, width = GLASS_INPUT_SIZE
    return transforms.Compose([
        transforms.Resize((height, width)),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_model(device, checkpoint_path, glass_dir):
    backbones, GLASS = load_glass_modules(glass_dir)
    height, width = GLASS_INPUT_SIZE

    backbone = backbones.load(GLASS_BACKBONE)
    model = GLASS(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=GLASS_LAYERS,
        device=device,
        input_shape=(3, height, width),
        pretrain_embed_dimension=GLASS_PRETRAIN_EMBED_DIM,
        target_embed_dimension=GLASS_TARGET_EMBED_DIM,
        patchsize=GLASS_PATCHSIZE,
        dsc_layers=GLASS_DSC_LAYERS,
        dsc_hidden=GLASS_DSC_HIDDEN,
        pre_proj=GLASS_PRE_PROJ,
    )

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "discriminator" in state_dict:
        model.discriminator.load_state_dict(state_dict["discriminator"])
        if "pre_projection" in state_dict and GLASS_PRE_PROJ > 0:
            model.pre_projection.load_state_dict(state_dict["pre_projection"])
    else:
        model.load_state_dict(state_dict, strict=False)

    return model


def build_sam_masker(device, enabled=True):
    if not enabled:
        return None
    try:
        from utils.preprocessing import SAMMasker

        masker = SAMMasker(device=device)
        print("[OK] SAM 마스커 로드 완료")
        return masker
    except Exception as exc:
        print(f"[WARN] SAM 마스커를 사용하지 않습니다: {exc}")
        return None


def apply_sam_mask(anomaly_map, original_np, sam_masker):
    if sam_masker is None:
        return anomaly_map

    _, obj_mask = sam_masker.mask_background(original_np)
    if obj_mask.shape != anomaly_map.shape:
        obj_mask = cv2.resize(
            obj_mask.astype("uint8"),
            (anomaly_map.shape[1], anomaly_map.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    anomaly_map = anomaly_map.copy()
    anomaly_map[~obj_mask] = 0
    return anomaly_map


def predict_path(model, transform, image_path, device):
    height, width = GLASS_INPUT_SIZE
    pil_img = Image.open(image_path).convert("RGB")
    original_np = np.array(pil_img.resize((width, height)))
    tensor = transform(pil_img).unsqueeze(0).to(device)
    scores, masks = model._predict(tensor)
    return float(scores[0]), np.array(masks[0]), original_np


def infer_single(model, transform, img_path, save_path, threshold, device, sam_masker=None):
    from utils.visual import save_result_image

    t0 = time.time()
    score, anomaly_map, original_np = predict_path(model, transform, img_path, device)
    infer_time = time.time() - t0
    anomaly_map = apply_sam_mask(anomaly_map, original_np, sam_masker)

    save_result_image(original_np, anomaly_map, save_path, threshold=threshold)
    print(
        f"  {os.path.basename(img_path):30s} "
        f"score={score:.4f} map_max={anomaly_map.max():.4f} "
        f"map_mean={anomaly_map.mean():.4f} infer={infer_time:.3f}s -> {save_path}"
    )


def collect_images(folder):
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    if not os.path.isdir(folder):
        print(f"[WARN] 폴더 없음: {folder}")
        return []
    return [
        os.path.join(folder, filename)
        for filename in sorted(os.listdir(folder))
        if filename.lower().endswith(valid_ext)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="단일 이미지 경로")
    parser.add_argument("--checkpoint", type=str, default=None, help="GLASS ckpt_best_*.pth 경로")
    parser.add_argument("--glass-dir", type=str, default=GLASS_PROJECT_DIR, help="외부 GLASS 프로젝트 폴더")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-sam", action="store_true", help="SAM 배경 마스킹 비활성화")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    checkpoint_path = find_checkpoint(args.checkpoint, args.glass_dir)
    model = build_model(device, checkpoint_path, args.glass_dir)
    transform = build_transform()

    print(f"[OK] GLASS 프로젝트: {args.glass_dir}")
    print(f"[OK] GLASS 체크포인트 로드: {checkpoint_path}")
    print(f"Device: {device}")

    sam_masker = build_sam_masker(device, enabled=not args.no_sam)
    os.makedirs(GLASS_RESULTS_DIR, exist_ok=True)

    if args.image:
        name = os.path.splitext(os.path.basename(args.image))[0]
        save_path = os.path.join(GLASS_RESULTS_DIR, f"{name}_result.png")
        infer_single(model, transform, args.image, save_path, args.threshold, device, sam_masker)
    else:
        for label, folder in [("bad", GLASS_TEST_BAD_DIR), ("good", GLASS_TEST_GOOD_DIR)]:
            files = collect_images(folder)
            print(f"\n[{label}] {len(files)}장 추론 중...")
            for img_path in files:
                name = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(GLASS_RESULTS_DIR, f"{label}_{name}_result.png")
                infer_single(model, transform, img_path, save_path, args.threshold, device, sam_masker)

    print(f"\n결과 저장 폴더: {GLASS_RESULTS_DIR}")


if __name__ == "__main__":
    main()
