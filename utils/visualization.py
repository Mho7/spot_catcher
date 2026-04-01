import numpy as np
import cv2
from PIL import Image


def create_heatmap_overlay(original_image, anomaly_map, threshold=0.5, alpha=0.4):
    h, w = original_image.shape[:2]

    if anomaly_map.shape != (h, w):
        anomaly_map = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap_bgr = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8),cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_rgb, alpha, 0)
    binary_mask = anomaly_map > threshold

    return overlay, binary_mask

def save_result_image(original_image, anomaly_map, save_path, threshold=0.5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import platform
    if platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    elif platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    overlay, binary_mask = create_heatmap_overlay(original_image, anomaly_map, threshold)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title('원본 이미지', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title('히트맵 오버레이', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(original_image)
    mask_overlay = np.zeros_like(original_image)
    mask_overlay[binary_mask] = [255, 0, 0]
    axes[2].imshow(mask_overlay, alpha=0.5)
    axes[2].set_title(f'이상 영역 (임계값: {threshold:.2f})', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def save_single_overlay(original_image, anomaly_map, save_path, threshold=0.5):
    """히트맵 오버레이 이미지만 단독으로 저장 (웹 UI용)"""
    overlay, _ = create_heatmap_overlay(original_image, anomaly_map, threshold, alpha=0.5)
    Image.fromarray(overlay).save(save_path)
    return save_path
