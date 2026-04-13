"""
단일 이미지 이상 탐지 결과 확인 스크립트

사용법:
    python inspect.py                          # data/test/bad/ 첫 번째 이미지
    python inspect.py data/test/bad/test1.png  # 특정 이미지 지정

pooling 설정은 config.py 에서 변경:
    USE_AGGREGATION = False           → 풀링 없음
    USE_AGGREGATION = True, AGGREGATION_KERNEL_SIZE = 1  → k=1
    USE_AGGREGATION = True, AGGREGATION_KERNEL_SIZE = 3  → k=3

결과 저장 위치:
    static/inspect_result.png
"""
import os
import sys
import numpy as np
from PIL import Image
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

from config import (
    TRAIN_DIR, TEST_BAD_DIR, STATIC_DIR, SAVE_DIR,
    IMAGE_SIZE, PATCHCORE_BACKBONE, PATCHCORE_LAYERS, CORESET_RATIO,
    USE_AGGREGATION, AGGREGATION_KERNEL_SIZE, ANOMALY_THRESHOLD
)
from models.patchcore import PatchCore
from utils.dataset import get_dataloader, get_default_transform

# 이미지 경로 결정
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    images = sorted([
        os.path.join(TEST_BAD_DIR, f) for f in os.listdir(TEST_BAD_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])
    if not images:
        print(f"[ERROR] 이미지 없음: {TEST_BAD_DIR}")
        sys.exit(1)
    img_path = images[0]

# config 설정에 맞는 pkl 파일명 결정
if not USE_AGGREGATION:
    pkl_name = "patchcore_no_pool.pkl"
else:
    pkl_name = f"patchcore_k{AGGREGATION_KERNEL_SIZE}.pkl"
pkl_path = os.path.join(SAVE_DIR, pkl_name)

# 모델 준비 (pkl 없으면 학습 후 저장)
model = PatchCore(
    backbone=PATCHCORE_BACKBONE,
    layers=PATCHCORE_LAYERS,
    coreset_ratio=CORESET_RATIO,
    use_aggregation=USE_AGGREGATION,
    aggregation_kernel_size=AGGREGATION_KERNEL_SIZE,
)

if os.path.exists(pkl_path):
    print(f"[LOAD] {pkl_name} 로드")
    model.load(pkl_path)
else:
    print(f"[TRAIN] {pkl_name} 없음 → 학습 시작")
    loader = get_dataloader(TRAIN_DIR, batch_size=4, shuffle=True, augment=True, repeat=20)
    if len(loader.dataset) == 0:
        print(f"[ERROR] 학습 이미지 없음: {TRAIN_DIR}")
        sys.exit(1)
    model.fit(loader)
    model.save(pkl_path)
    print(f"[SAVE] {pkl_path}")

# 추론
transform = get_default_transform()
pil = Image.open(img_path).convert('RGB')
tensor = transform(pil)
score, anomaly_map = model.predict(tensor)
original = np.array(pil.resize(IMAGE_SIZE))

detected = score >= ANOMALY_THRESHOLD
status = "이상 탐지" if detected else "정상 판정"
color = '#d62728' if detected else '#2ca02c'

agg_label = f"k={AGGREGATION_KERNEL_SIZE}" if USE_AGGREGATION else "풀링 없음"
print(f"설정: {agg_label} | score: {score:.4f} | {status}")

# 시각화: [원본] [히트맵]
# anomaly_map을 원본 크기로 리사이즈
anomaly_map_resized = np.array(
    Image.fromarray((anomaly_map * 255).astype(np.uint8)).resize(IMAGE_SIZE, Image.BILINEAR)
) / 255.0
heatmap_rgb = (plt.cm.hot(anomaly_map_resized)[:, :, :3] * 255).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(9, 5))

axes[0].imshow(original)
axes[0].set_title("원본", fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(heatmap_rgb)
plt.colorbar(plt.cm.ScalarMappable(cmap='hot'), ax=axes[1], fraction=0.046, pad=0.04)
axes[1].set_title("히트맵", fontsize=12, fontweight='bold')
axes[1].axis('off')

filename = os.path.basename(img_path)
fig.suptitle(
    f"{filename}  |  {agg_label}  |  score: {score:.4f}  |  {status}",
    fontsize=12, fontweight='bold', color=color
)
plt.tight_layout()

save_path = os.path.join(STATIC_DIR, "inspect_result.png")
plt.savefig(save_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"저장: {save_path}")
