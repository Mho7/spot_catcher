"""
pooling 설정별 결함 식별도 비교 (CMD 출력)

실행:  python score_compare.py
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
    TRAIN_DIR, TEST_BAD_DIR, SAVE_DIR, STATIC_DIR,
    IMAGE_SIZE, PATCHCORE_BACKBONE, PATCHCORE_LAYERS, CORESET_RATIO,
)
from models.patchcore import PatchCore
from utils.dataset import get_dataloader, get_default_transform

CONFIGS = [
    {"label": "no pool", "use_agg": False, "kernel": 1, "pkl": "patchcore_no_pool.pkl"},
    {"label": "k = 1",   "use_agg": True,  "kernel": 1, "pkl": "patchcore_k1.pkl"},
    {"label": "k = 3",   "use_agg": True,  "kernel": 3, "pkl": "patchcore_k3.pkl"},
    {"label": "k = 5",   "use_agg": True,  "kernel": 5, "pkl": "patchcore_k5.pkl"},
]


def load_or_train(cfg):
    import io, contextlib
    pkl_path = os.path.join(SAVE_DIR, cfg["pkl"])
    model = PatchCore(
        backbone=PATCHCORE_BACKBONE,
        layers=PATCHCORE_LAYERS,
        coreset_ratio=CORESET_RATIO,
        use_aggregation=cfg["use_agg"],
        aggregation_kernel_size=cfg["kernel"],
    )
    if os.path.exists(pkl_path):
        with contextlib.redirect_stdout(io.StringIO()):
            model.load(pkl_path)
    else:
        loader = get_dataloader(TRAIN_DIR, batch_size=4, shuffle=True, augment=True, repeat=20)
        model.fit(loader)
        model.save(pkl_path)
    return model


# 이미지 로드
bad_path = os.path.join(TEST_BAD_DIR, "synthetic_hair.png")
if not os.path.exists(bad_path):
    print("[ERROR] synthetic_hair.png 없음 → 먼저 python make_synthetic.py 실행")
    sys.exit(1)

good_path = sorted([
    os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
])[0]

transform = get_default_transform()
bad_pil  = Image.open(bad_path).convert('RGB')
good_pil = Image.open(good_path).convert('RGB')

original = np.array(bad_pil.resize(IMAGE_SIZE))

# 모델 로드 & 스코어 계산
import io, contextlib
results = []
for cfg in CONFIGS:
    model = load_or_train(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        good_score, _ = model.predict(transform(good_pil))
        bad_score, anomaly_map = model.predict(transform(bad_pil))
    disc = bad_score - good_score
    amap = np.array(
        Image.fromarray((anomaly_map * 255).astype(np.uint8)).resize(IMAGE_SIZE, Image.BILINEAR)
    ) / 255.0
    results.append({"label": cfg["label"], "good": good_score, "bad": bad_score, "disc": disc, "amap": amap})

best = max(results, key=lambda r: r["disc"])

C1, C2, C3, C4 = 10, 13, 13, 11
head  = f"  {'설정':<{C1}} {'정상 score':>{C2}} {'결함 score':>{C3}} {'식별도':>{C4}}"
div   = f"  {'─'*C1}  {'─'*C2}  {'─'*C3}  {'─'*C4}"

print()
print(f"  POOLING  식별도 분석")
print(f"  {'─' * (C1+C2+C3+C4+6)}")
print(head)
print(div)
for res in results:
    marker = "  ◀" if res["label"] == best["label"] else ""
    print(f"  {res['label']:<{C1}} {res['good']:>{C2}.4f}  {res['bad']:>{C3}.4f}  {res['disc']:>{C4}.4f}{marker}")
print(f"  {'─' * (C1+C2+C3+C4+6)}")
print(f"  최적  {best['label']}   식별도 {best['disc']:.4f}")
print()

# 이미지 저장: 원본 + 각 pooling별 anomaly map
n = len(results)
fig, axes = plt.subplots(1, n + 1, figsize=(3.8 * (n + 1), 4.5), facecolor='white')

axes[0].imshow(original)
axes[0].set_title("원본", fontsize=11, fontweight='bold')
axes[0].axis('off')

for i, res in enumerate(results):
    ax = axes[i + 1]
    ax.imshow(original)
    ax.set_title(res["label"], fontsize=11,
                 fontweight='bold' if res["label"] == best["label"] else 'normal')
    ax.set_xlabel(f"식별도 {res['disc']:.4f}", fontsize=9,
                  color='#1a1a2e' if res["label"] == best["label"] else '#888888')
    ax.axis('off')
    if res["label"] == best["label"]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#1a1a2e')
            spine.set_linewidth(2)

plt.tight_layout()
save_path = os.path.join(STATIC_DIR, "score_compare.png")
plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor='white')
plt.close()
print(f"이미지 저장: {save_path}")
