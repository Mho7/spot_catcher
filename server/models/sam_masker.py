"""
SAM 기반 배경 마스킹 유틸.

객체 마스크를 만든 뒤 모폴로지 정리 + erosion 으로 경계 영역까지 보호하고,
그 외 영역의 anomaly_map 값을 0 으로 죽이는 데 사용합니다.
"""
import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from config import SAM_MODEL_PATH, SAM_MODEL_TYPE


class SAMMasker:
    def __init__(self, model_path=SAM_MODEL_PATH, model_type=SAM_MODEL_TYPE, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device)
        self.predictor = SamPredictor(sam)

    def mask_background(self, image_np, bbox=None):
        """이미지에서 객체 마스크를 만들어 배경을 검정으로 채워 반환."""
        h, w = image_np.shape[:2]
        self.predictor.set_image(image_np)

        if bbox is None:
            bbox = np.array([0, 0, w, h])
        else:
            bbox = np.array(bbox)

        masks, scores, _ = self.predictor.predict(box=bbox, multimask_output=True)
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]

        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        if not mask[cy, cx]:
            mask = ~mask

        mask_uint8 = mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        mask_uint8 = cv2.erode(mask_uint8, erode_kernel, iterations=1)
        mask = mask_uint8 > 0

        masked = image_np.copy()
        masked[~mask] = 0
        return masked, mask

    def apply_to_anomaly_map(self, anomaly_map, original_np):
        """anomaly_map 의 객체 외부 영역을 0 으로 만들어 반환."""
        _, obj_mask = self.mask_background(original_np)
        if obj_mask.shape != anomaly_map.shape:
            obj_mask = cv2.resize(
                obj_mask.astype(np.uint8),
                (anomaly_map.shape[1], anomaly_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        masked_map = anomaly_map.copy()
        masked_map[~obj_mask] = 0
        return masked_map, obj_mask
