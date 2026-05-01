"""
전처리 유틸리티: SAM 마스킹 + 크롭

파이프라인:
1. SAM으로 객체 세그멘테이션 → 배경 마스킹 (검정)
2. 마스크 바운딩박스로 크롭
"""
import os
import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAM_MODEL_PATH, SAM_MODEL_TYPE


class SAMMasker:
    """
    SAM을 이용한 배경 마스킹

    사용법:
        masker = SAMMasker()
        masked_image = masker.mask_background(cropped_image)
    """

    def __init__(self, model_path=SAM_MODEL_PATH, model_type=SAM_MODEL_TYPE,
                 device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device)
        self.predictor = SamPredictor(sam)

    def mask_background(self, image_np, bbox=None):
        """
        이미지에서 객체를 세그멘테이션하고 배경을 검정으로 마스킹

        Args:
            image_np: numpy 배열 (H, W, 3) RGB
            bbox: (x1, y1, x2, y2) 바운딩박스. None이면 이미지 전체를 박스로 사용

        Returns:
            masked: 배경이 검정인 이미지 (numpy 배열)
            mask: 바이너리 마스크 (H, W) bool
        """
        h, w = image_np.shape[:2]
        self.predictor.set_image(image_np)

        # 바운딩박스 프롬프트 (크롭 후이므로 이미지 전체가 객체 영역)
        if bbox is None:
            bbox = np.array([0, 0, w, h])
        else:
            bbox = np.array(bbox)

        masks, scores, _ = self.predictor.predict(
            box=bbox,
            multimask_output=True,
        )

        # 가장 높은 점수의 마스크 선택
        best_idx = np.argmax(scores)
        mask = masks[best_idx]  # (H, W) bool

        # 마스크 검증: 바운딩박스 중심이 마스크에 포함되어야 함
        # 포함되지 않으면 SAM이 전경/배경을 반전한 것이므로 마스크를 뒤집음
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        if not mask[cy, cx]:
            mask = ~mask

        # 모폴로지로 마스크 노이즈 제거 (경계 산란 픽셀 제거)
        mask_uint8 = mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask = mask_uint8 > 0

        # 배경을 검정으로
        masked = image_np.copy()
        masked[~mask] = 0

        return masked, mask


def preprocess_image(image_np, sam_masker):
    """
    전처리 파이프라인: SAM 마스킹만 수행 (크롭 없음)

    Args:
        image_np: 원본 이미지 numpy 배열 (H, W, 3) RGB
        sam_masker: SAMMasker 인스턴스

    Returns:
        masked: 배경이 검정인 이미지 (numpy 배열, 원본 크기 유지)
        mask: 객체 마스크 (H, W) bool
    """
    masked, mask = sam_masker.mask_background(image_np)
    return masked, mask


def preprocess_directory(input_dir, output_dir, sam_masker):
    """
    폴더 내 모든 이미지를 마스킹 처리하여 저장

    Args:
        input_dir: 원본 이미지 폴더
        output_dir: 전처리된 이미지 저장 폴더
        sam_masker: SAMMasker 인스턴스
    """
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    image_files = [f for f in sorted(os.listdir(input_dir))
                   if f.lower().endswith(valid_extensions)]

    print(f"[전처리] {len(image_files)}개 이미지 처리 시작...")

    for i, filename in enumerate(image_files):
        img_path = os.path.join(input_dir, filename)
        image = np.array(Image.open(img_path).convert('RGB'))

        masked, _ = preprocess_image(image, sam_masker)

        # 저장
        save_path = os.path.join(output_dir, filename)
        Image.fromarray(masked).save(save_path)
        print(f"  [{i+1}/{len(image_files)}] {filename} → 완료")

    print(f"[전처리] 완료! 저장 위치: {output_dir}")
