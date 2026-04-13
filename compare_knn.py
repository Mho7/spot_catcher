"""
k-NN 비교 스크립트

각 패치의 이상 점수를 메모리 뱅크에서
가장 가까운 k개 이웃의 평균 거리로 계산.

k=1  → 가장 가까운 1개 (기본값)
k=3  → 가장 가까운 3개 평균
k=5  → 가장 가까운 5개 평균
k=7  → 가장 가까운 7개 평균

실행:  python compare_knn.py
"""
import os
import sys
import io
import contextlib
from PIL import Image

from config import (
    TRAIN_DIR, TEST_BAD_DIR, SAVE_DIR,
    IMAGE_SIZE, PATCHCORE_BACKBONE, PATCHCORE_LAYERS, CORESET_RATIO,
)
from models.patchcore import PatchCore
from utils.dataset import get_dataloader, get_default_transform

KNN_VALUES = [1, 3, 5, 7]

# 공통 모델 1개 사용 (pooling 없음, 순수 k-NN 효과만 비교)
PKL_PATH = os.path.join(SAVE_DIR, "patchcore_no_pool.pkl")


def load_or_train():
    model = PatchCore(
        backbone=PATCHCORE_BACKBONE,
        layers=PATCHCORE_LAYERS,
        coreset_ratio=CORESET_RATIO,
        use_aggregation=False,
        aggregation_kernel_size=1,
    )
    if os.path.exists(PKL_PATH):
        with contextlib.redirect_stdout(io.StringIO()):
            model.load(PKL_PATH)
    else:
        print("  학습 시작...")
        loader = get_dataloader(TRAIN_DIR, batch_size=4, shuffle=True, augment=True, repeat=20)
        model.fit(loader)
        model.save(PKL_PATH)
    return model


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

model = load_or_train()

results = []
for k in KNN_VALUES:
    good_score, _ = model.predict(transform(good_pil), knn_k=k)
    bad_score,  _ = model.predict(transform(bad_pil),  knn_k=k)
    disc = bad_score - good_score
    results.append({"k": k, "good": good_score, "bad": bad_score, "disc": disc})

best = max(results, key=lambda r: r["disc"])

C1, C2, C3, C4 = 8, 13, 13, 11
head = f"  {'k-NN':<{C1}} {'정상 score':>{C2}} {'결함 score':>{C3}} {'식별도':>{C4}}"
div  = f"  {'─'*C1}  {'─'*C2}  {'─'*C3}  {'─'*C4}"

print()
print("  k-NN 비교  (pooling 없음 고정)")
print(f"  {'─' * (C1+C2+C3+C4+6)}")
print(head)
print(div)
for res in results:
    marker = "  ◀" if res["k"] == best["k"] else ""
    print(f"  {'k='+str(res['k']):<{C1}} {res['good']:>{C2}.4f}  {res['bad']:>{C3}.4f}  {res['disc']:>{C4}.4f}{marker}")
print(f"  {'─' * (C1+C2+C3+C4+6)}")
print(f"  최적: k={best['k']}  (식별도 {best['disc']:.4f})")
print()
