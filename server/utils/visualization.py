import os
import sys
import numpy as np
import cv2
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OVERLAY_THRESHOLD


def make_heatmap(original_image, anomaly_map):
    """anomaly_map을 JET 컬러맵으로 변환한 순수 히트맵 이미지 반환 (numpy RGB array)"""
    h, w = original_image.shape[:2]
    if anomaly_map.shape != (h, w):
        anomaly_map = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)

    scaled = (np.clip(anomaly_map, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def save_heatmap(original_image, anomaly_map, save_path):
    Image.fromarray(make_heatmap(original_image, anomaly_map)).save(save_path)
    return save_path


def make_single_overlay(original_image, anomaly_map, threshold=OVERLAY_THRESHOLD):
    """빨간 마스킹 오버레이 이미지 생성 (numpy array 반환)"""
    h, w = original_image.shape[:2]
    if anomaly_map.shape != (h, w):
        anomaly_map = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
    binary_mask = anomaly_map > threshold
    red_layer = np.zeros_like(original_image)
    red_layer[binary_mask] = [255, 0, 0]
    return cv2.addWeighted(original_image, 1.0, red_layer, 0.5, 0)


def save_single_overlay(original_image, anomaly_map, save_path, threshold=OVERLAY_THRESHOLD):
    overlay = make_single_overlay(original_image, anomaly_map, threshold)
    Image.fromarray(overlay).save(save_path)
    return save_path
