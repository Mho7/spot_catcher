"""
풀링 커널 크기 비교 스크립트

실행 방법:
    python compare_pooling.py

이 스크립트가 하는 일:
    1. data/train/good/ 의 정상 이미지로 커널 크기별 PatchCore 각각 학습
    2. data/test/bad/ 의 결함 이미지로 각 모델 추론
    3. 커널별 히트맵 비교 이미지 저장 → static/pooling_compare/
    4. 수치 비교 테이블 출력 (anomaly_score, 탐지 여부)
    5. AUROC 비교 (data/test/good/ 이미지도 있을 경우)

커널 크기 의미:
    1  → 풀링 없음 (원시 패치, 노이즈 민감, 미세 결함 유리)
    3  → 3x3 주변 평균 (기본값, 균형)
    5  → 5x5 주변 평균 (더 부드러움, 작은 결함 놓칠 수 있음)
    7  → 7x7 주변 평균 (가장 뭉개짐, 큰 결함만 탐지)
"""
import os
import sys
import numpy as np
from PIL import Image
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

from config import (
    TRAIN_DIR, TEST_GOOD_DIR, TEST_BAD_DIR, STATIC_DIR,
    IMAGE_SIZE, PATCHCORE_BACKBONE, PATCHCORE_LAYERS, CORESET_RATIO
)
from models.patchcore import PatchCore
from utils.dataset import get_dataloader, get_default_transform

# ============================================
# 테스트할 커널 크기 목록
# ============================================
KERNEL_SIZES = [1, 3, 5, 7]

OUTPUT_DIR = os.path.join(STATIC_DIR, "pooling_compare")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_model(kernel_size):
    """지정한 커널 크기로 PatchCore 학습 후 반환"""
    use_agg = kernel_size > 1
    model = PatchCore(
        backbone=PATCHCORE_BACKBONE,
        layers=PATCHCORE_LAYERS,
        coreset_ratio=CORESET_RATIO,
        use_aggregation=use_agg,
        aggregation_kernel_size=kernel_size,
    )
    loader = get_dataloader(TRAIN_DIR, batch_size=4, shuffle=True,
                            augment=True, repeat=20)
    if len(loader.dataset) == 0:
        print(f"[ERROR] 학습 이미지 없음: {TRAIN_DIR}")
        sys.exit(1)
    model.fit(loader)
    return model


def collect_images(folder):
    """폴더에서 이미지 경로 리스트 반환"""
    if not os.path.exists(folder):
        return []
    valid = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(valid)
    ]


def run_inference(model, img_path):
    transform = get_default_transform()
    pil = Image.open(img_path).convert('RGB')
    tensor = transform(pil)
    score, anomaly_map = model.predict(tensor)
    original = np.array(pil.resize(IMAGE_SIZE))
    return score, anomaly_map, original


def save_comparison_image(img_path, models_dict, threshold, save_path):
    """
    한 이미지에 대해 커널별 히트맵을 나란히 저장

    레이아웃:
        [원본] [kernel=1] [kernel=3] [kernel=5] [kernel=7]
        각 히트맵 아래에 score 표시, 탐지 여부(O/X) 표시
    """
    n_cols = 1 + len(KERNEL_SIZES)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))

    transform = get_default_transform()
    pil = Image.open(img_path).convert('RGB')
    original = np.array(pil.resize(IMAGE_SIZE))

    # 원본 이미지
    axes[0].imshow(original)
    axes[0].set_title("원본", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 커널별 히트맵
    for i, k in enumerate(KERNEL_SIZES):
        ax = axes[i + 1]
        model = models_dict[k]
        score, anomaly_map, _ = run_inference(model, img_path)

        detected = score >= threshold
        color = '#d62728' if detected else '#2ca02c'  # 빨강=이상, 초록=정상 판정
        status = "이상 탐지" if detected else "정상 판정"

        # 원본 위에 히트맵 오버레이
        ax.imshow(original)
        im = ax.imshow(anomaly_map, cmap='hot', alpha=0.55, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        title = f"kernel={k}" if k > 1 else "kernel=1\n(풀링 없음)"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"score: {score:.4f}\n{status}",
                      fontsize=10, color=color, fontweight='bold')
        ax.axis('off')

    filename = os.path.basename(img_path)
    fig.suptitle(f"풀링 비교: {filename}", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def print_score_table(results):
    """
    results: { img_name: { kernel_size: score } }
    """
    col_w = 12
    header = f"{'이미지':<28}" + "".join(f"{'k='+str(k):>{col_w}}" for k in KERNEL_SIZES)
    print("\n" + "=" * (28 + col_w * len(KERNEL_SIZES)))
    print("  anomaly_score 비교 (높을수록 결함 의심)")
    print("=" * (28 + col_w * len(KERNEL_SIZES)))
    print(header)
    print("-" * (28 + col_w * len(KERNEL_SIZES)))
    for img_name, scores in results.items():
        row = f"{img_name[:27]:<28}"
        row += "".join(f"{scores[k]:>{col_w}.4f}" for k in KERNEL_SIZES)
        print(row)
    print("=" * (28 + col_w * len(KERNEL_SIZES)))


def compute_auroc_per_kernel(models_dict, good_paths, bad_paths):
    """good/bad 이미지가 모두 있을 때 커널별 AUROC 계산"""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None

    print("\n[INFO] AUROC 계산 중...")
    auroc_results = {}
    for k, model in models_dict.items():
        labels, scores = [], []
        for path in good_paths:
            s, _, _ = run_inference(model, path)
            labels.append(0); scores.append(s)
        for path in bad_paths:
            s, _, _ = run_inference(model, path)
            labels.append(1); scores.append(s)
        auroc = roc_auc_score(labels, scores)
        auroc_results[k] = auroc
    return auroc_results


def save_auroc_bar(auroc_results):
    """AUROC 막대그래프 저장"""
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = [str(k) for k in KERNEL_SIZES]
    vals = [auroc_results[k] for k in KERNEL_SIZES]
    colors = ['#1f77b4' if v == max(vals) else '#aec7e8' for v in vals]
    bars = ax.bar(keys, vals, color=colors, edgecolor='black', width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("풀링 커널 크기", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("커널 크기별 AUROC 비교", fontsize=13, fontweight='bold')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "auroc_comparison.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  AUROC 그래프 저장: {save_path}")


def main():
    print("=" * 60)
    print("  풀링 커널 크기 비교 실험")
    print(f"  테스트 커널: {KERNEL_SIZES}")
    print("=" * 60)

    # ----------------------------------------
    # 1. 커널별 모델 학습
    # ----------------------------------------
    print("\n[STEP 1] 커널별 PatchCore 학습")
    models_dict = {}
    for k in KERNEL_SIZES:
        label = f"kernel={k}" if k > 1 else "kernel=1 (풀링 없음)"
        print(f"\n--- {label} ---")
        models_dict[k] = build_model(k)

    # ----------------------------------------
    # 2. 결함 이미지 히트맵 비교
    # ----------------------------------------
    bad_paths = collect_images(TEST_BAD_DIR)
    good_paths = collect_images(TEST_GOOD_DIR)

    if not bad_paths:
        print(f"\n[WARN] 결함 이미지 없음: {TEST_BAD_DIR}")
        print("  히트맵 비교를 건너뜁니다.")
    else:
        print(f"\n[STEP 2] 결함 이미지 히트맵 비교 ({len(bad_paths)}장)")

        # 탐지 여부 기준: 정상 이미지 평균 score의 2배 (or 0.5 고정)
        threshold = 0.5

        score_table = {}
        for img_path in bad_paths:
            fname = os.path.basename(img_path)
            print(f"  처리 중: {fname}")

            scores = {}
            for k, model in models_dict.items():
                s, _, _ = run_inference(model, img_path)
                scores[k] = s
            score_table[fname] = scores

            save_path = os.path.join(OUTPUT_DIR, f"compare_{fname}")
            save_comparison_image(img_path, models_dict, threshold, save_path)
            print(f"    저장: {save_path}")

        print_score_table(score_table)

    # ----------------------------------------
    # 3. AUROC 비교 (good + bad 모두 있을 때)
    # ----------------------------------------
    if good_paths and bad_paths:
        print(f"\n[STEP 3] AUROC 비교 (정상 {len(good_paths)}장 / 결함 {len(bad_paths)}장)")
        auroc_results = compute_auroc_per_kernel(models_dict, good_paths, bad_paths)
        if auroc_results:
            print("\n  커널별 AUROC:")
            best_k = max(auroc_results, key=auroc_results.get)
            for k, auroc in auroc_results.items():
                mark = " ★ 최고" if k == best_k else ""
                print(f"    kernel={k}: {auroc:.4f}{mark}")
            save_auroc_bar(auroc_results)
    else:
        print("\n[INFO] good/bad 이미지 모두 있어야 AUROC 계산 가능 (현재 생략)")

    print(f"\n[완료] 결과 저장 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
