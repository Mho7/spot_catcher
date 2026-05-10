"""
GLASS 체크포인트(ckpt_best_9.pth)로 추론 → save_result_image 시각화

사용법:
    python infer_glass.py
    python infer_glass.py --image data/glass_format/spot/test/bad/frame_0051.png
    python infer_glass.py --threshold 0.5
"""
import os
import sys
import argparse
import glob
import time
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

# spot_catcher의 utils를 먼저 import (GLASS의 utils와 충돌 방지)
from utils.visual import save_result_image
from utils.preprocessing import SAMMasker

# GLASS 프로젝트 모듈 import
GLASS_DIR = r"C:\Users\User\github\GLASS"
sys.path.insert(0, GLASS_DIR)

import backbones
from glass import GLASS

# run-spot.sh 기준 설정
CKPT_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "ckpt_best_186.pth")
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


class CLAHE:
    """CLAHE 전처리 transform (PIL Image → PIL Image)"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        from PIL import Image as _Image
        return _Image.fromarray(result)


def build_model(device):
    backbone = backbones.load(BACKBONE_NAME)
    model = GLASS(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=LAYERS,
        device=device,
        input_shape=(3, IMAGESIZE_H, IMAGESIZE_W),
        pretrain_embed_dimension=PRETRAIN_EMBED_DIM,
        target_embed_dimension=TARGET_EMBED_DIM,
        patchsize=PATCHSIZE,
        dsc_layers=DSC_LAYERS,
        dsc_hidden=DSC_HIDDEN,
        pre_proj=PRE_PROJ,
    )

    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.discriminator.load_state_dict(state_dict["discriminator"])
    if "pre_projection" in state_dict:
        model.pre_projection.load_state_dict(state_dict["pre_projection"])
    print(f"[OK] GLASS 체크포인트 로드: {CKPT_PATH}")
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGESIZE_H, IMAGESIZE_W)),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def infer_single(model, img_path, save_path, threshold, device, sam_masker=None):
    transform = get_transform()

    pil_img = Image.open(img_path).convert("RGB")
    original_np = np.array(pil_img.resize((IMAGESIZE_W, IMAGESIZE_H)))

    tensor = transform(pil_img).unsqueeze(0).to(device)

    t0 = time.time()
    scores, masks = model._predict(tensor)
    infer_time = time.time() - t0

    anomaly_map = masks[0]  # (H, W) numpy array

    # SAM 마스크로 배경+경계 영역의 anomaly score를 0으로 제거
    if sam_masker is not None:
        _, obj_mask = sam_masker.mask_background(original_np)
        if obj_mask.shape != anomaly_map.shape:
            obj_mask = cv2.resize(obj_mask.astype(np.uint8),
                                  (anomaly_map.shape[1], anomaly_map.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        anomaly_map[~obj_mask] = 0

    save_result_image(original_np, anomaly_map, save_path, threshold=threshold)
    print(f"  {os.path.basename(img_path):30s} score={scores[0]:.4f}  map_max={anomaly_map.max():.4f}  map_mean={anomaly_map.mean():.4f}  infer={infer_time:.3f}s → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="단일 이미지 경로")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(device)
    sam_masker = SAMMasker(device=device)
    print("[OK] SAM 마스커 로드 완료")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.image:
        name = os.path.splitext(os.path.basename(args.image))[0]
        save_path = os.path.join(OUTPUT_DIR, f"{name}_result.png")
        infer_single(model, args.image, save_path, args.threshold, device, sam_masker)
    else:
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        for label, folder in [("bad", TEST_BAD_DIR), ("good", TEST_GOOD_DIR)]:
            if not os.path.isdir(folder):
                print(f"[WARN] 폴더 없음: {folder}")
                continue
            files = sorted(f for f in os.listdir(folder) if f.lower().endswith(valid_ext))
            print(f"\n[{label}] {len(files)}장 추론 중...")
            for f in files:
                img_path = os.path.join(folder, f)
                name = os.path.splitext(f)[0]
                save_path = os.path.join(OUTPUT_DIR, f"{label}_{name}_result.png")
                infer_single(model, img_path, save_path, args.threshold, device, sam_masker)

    print(f"\n결과 저장 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
