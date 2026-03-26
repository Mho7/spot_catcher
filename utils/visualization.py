"""
결함 시각화 유틸리티

역할:
- 이상 점수 맵을 히트맵으로 변환
- 원본 이미지 위에 히트맵 오버레이
- 결과 이미지 저장k
"""
import numpy as np
import cv2
from PIL import Image


def create_heatmap_overlay(original_image, anomaly_map, threshold=0.5, alpha=0.4):
    """
    원본 이미지 위에 이상 점수 히트맵을 오버레이

    Args:
        original_image: 원본 이미지 (numpy array, [H, W, 3], 0~255, RGB)
        anomaly_map: 이상 점수 맵 (numpy array, [H, W], 0~1 범위)
        threshold: 이상 판단 임계값
        alpha: 히트맵 투명도 (0=투명, 1=불투명)

    Returns:
        overlay: 히트맵이 오버레이된 이미지 (numpy array, RGB)
        binary_mask: 이상 영역 마스크 (True/False)
    """
    h, w = original_image.shape[:2]

    # anomaly_map을 원본 이미지 크기로 리사이즈 (w, h 순서 주의)
    if anomaly_map.shape != (h, w):
        anomaly_map = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # 히트맵 생성 (COLORMAP_JET: 파란색=정상, 빨간색=이상)
    # applyColorMap은 BGR 출력 → RGB로 변환
    heatmap_bgr = cv2.applyColorMap(
        (anomaly_map * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 원본 이미지(RGB)와 히트맵(RGB) 오버레이
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_rgb, alpha, 0)

    # 이상 영역 마스크 생성
    binary_mask = anomaly_map > threshold

    return overlay, binary_mask


def save_result_image(original_image, anomaly_map, save_path, threshold=0.5):
    """
    탐지 결과를 3열 비교 이미지로 저장
    [원본 | 히트맵 오버레이 | 이상 영역 마스크]

    Args:
        original_image: 원본 이미지 (numpy, [H,W,3], 0~255)
        anomaly_map: 이상 점수 맵 (numpy, [H,W])
        save_path: 저장 경로
        threshold: 이상 판단 임계값
    """
    # matplotlib은 이 함수에서만 사용하므로 lazy import
    import matplotlib
    matplotlib.use('Agg')  # GUI 없이 사용 (서버 환경)
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 한글 폰트 설정 (macOS: AppleGothic, Windows: Malgun Gothic)
    import platform
    if platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    elif platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    overlay, binary_mask = create_heatmap_overlay(
        original_image, anomaly_map, threshold
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 원본 이미지
    axes[0].imshow(original_image)
    axes[0].set_title('원본 이미지', fontsize=14)
    axes[0].axis('off')

    # 히트맵 오버레이
    axes[1].imshow(overlay)
    axes[1].set_title('히트맵 오버레이', fontsize=14)
    axes[1].axis('off')

    # 이상 영역 마스크
    axes[2].imshow(original_image)
    mask_overlay = np.zeros_like(original_image)
    mask_overlay[binary_mask] = [255, 0, 0]  # 빨간색으로 이상 영역 강조
    axes[2].imshow(mask_overlay, alpha=0.5)
    axes[2].set_title(f'이상 영역 (임계값: {threshold:.2f})', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"  결과 이미지 저장: {save_path}")


def save_single_overlay(original_image, anomaly_map, save_path, threshold=0.5):
    """
    히트맵 오버레이 이미지만 단독으로 저장 (웹 UI용)
    """
    overlay, _ = create_heatmap_overlay(
        original_image, anomaly_map, threshold, alpha=0.5
    )

    Image.fromarray(overlay).save(save_path)
    return save_path
