import numpy as np
import cv2
from PIL import Image


def make_single_overlay(original_image, anomaly_map, threshold=0.5):
    """
    빨간 마스킹 오버레이 이미지 생성 (numpy array 반환)
    이상 영역을 빨간색으로 반투명하게 강조
    """
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
