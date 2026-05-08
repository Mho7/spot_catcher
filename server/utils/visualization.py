import numpy as np
import cv2
from PIL import Image


def make_heatmap(original_image, anomaly_map):
    """anomaly_map을 HOT 컬러맵으로 변환한 순수 히트맵 이미지 반환 (numpy RGB array)"""
    h, w = original_image.shape[:2]
    if anomaly_map.shape != (h, w):
        anomaly_map = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)

    norm = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def save_heatmap(original_image, anomaly_map, save_path):
    Image.fromarray(make_heatmap(original_image, anomaly_map)).save(save_path)
    return save_path


def make_single_overlay(original_image, anomaly_map, threshold=0.5):
    h, w = original_image.shape[:2]
    if anomaly_map.shape != (h, w):
        anomaly_map = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
    binary_mask = anomaly_map > threshold
    red_layer = np.zeros_like(original_image)
    red_layer[binary_mask] = [255, 0, 0]
    return cv2.addWeighted(original_image, 1.0, red_layer, 0.5, 0)


def save_single_overlay(original_image, anomaly_map, save_path, threshold=0.5):
    overlay = make_single_overlay(original_image, anomaly_map, threshold)
    Image.fromarray(overlay).save(save_path)
    return save_path
