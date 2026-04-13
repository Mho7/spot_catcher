"""
풀링 방식 비교 스크립트

비교 대상 (총 3가지):
    no_pool  → avg_pool2d 없음 (원시 패치, 미세 결함 유리)
    k=1      → avg_pool2d kernel=1 (풀링 연산은 하되 1x1, 사실상 no_pool과 유사)
    k=3      → avg_pool2d kernel=3 (기본값, 균형) ← saved_models/patchcore.pkl 재사용

실행 방법:
    python compare_pooling.py

결과 위치:
    static/pooling_compare/compare_<이미지명>.png
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
    TRAIN_DIR, TEST_GOOD_DIR, TEST_BAD_DIR, STATIC_DIR, SAVE_DIR,
    IMAGE_SIZE, PATCHCORE_BACKBONE, PATCHCORE_LAYERS, CORESET_RATIO
)
from models.patchcore import PatchCore
from utils.dataset import get_dataloader, get_default_transform

# ============================================
# 비교할 모델 설정
# pkl_path 가 있으면 학습 건너뛰고 로드
# ============================================
CONFIGS = [
    {
        "label":       "no_pool",
        "use_agg":     False,
        "kernel_size": 1,
        "pkl_path":    os.path.join(SAVE_DIR, "patchcore_no_pool.pkl"),
    },
    {
        "label":       "k=1",
        "use_agg":     True,
        "kernel_size": 1,
        "pkl_path":    os.path.join(SAVE_DIR, "patchcore_k1.pkl"),
    },
    {
        "label":       "k=3",
        "use_agg":     True,
        "kernel_size": 3,
        "pkl_path":    os.path.join(SAVE_DIR, "patchcore.pkl"),  # 기존 재사용
    },
]

OUTPUT_DIR = os.path.join(STATIC_DIR, "pooling_compare")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_model(cfg):
    """cfg에 따라 PatchCore 로드 또는 학습"""
    model = PatchCore(
        backbone=PATCHCORE_BACKBONE,
        layers=PATCHCORE_LAYERS,
        coreset_ratio=CORESET_RATIO,
        use_aggregation=cfg["use_agg"],
        aggregation_kernel_size=cfg["kernel_size"],
    )
    if cfg["pkl_path"] and os.path.exists(cfg["pkl_path"]):
        print(f"  [LOAD] {cfg['pkl_path']} 재사용")
        model.load(cfg["pkl_path"])
    else:
        loader = get_dataloader(TRAIN_DIR, batch_size=4, shuffle=True,
                                augment=True, repeat=20)
        if len(loader.dataset) == 0:
            print(f"[ERROR] 학습 이미지 없음: {TRAIN_DIR}")
            sys.exit(1)
        model.fit(loader)
        if cfg["pkl_path"]:
            model.save(cfg["pkl_path"])
            print(f"  [SAVE] {cfg['pkl_path']} 저장 완료")
    return model


def collect_images(folder):
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
    2행 레이아웃:
      행1: 원본 | no_pool 히트맵 | k=1 히트맵 | k=3 히트맵
      행2: (빈칸) | 원본+오버레이 | 원본+오버레이 | 원본+오버레이
    """
    n_configs = len(CONFIGS)
    n_cols = 1 + n_configs
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 9))

    pil = Image.open(img_path).convert('RGB')
    original = np.array(pil.resize(IMAGE_SIZE))

    # 행1 첫 칸: 원본
    axes[0][0].imshow(original)
    axes[0][0].set_title("원본", fontsize=12, fontweight='bold')
    axes[0][0].axis('off')

    # 행2 첫 칸: 비워둠
    axes[1][0].axis('off')

    for i, cfg in enumerate(CONFIGS):
        model = models_dict[cfg["label"]]
        score, anomaly_map, _ = run_inference(model, img_path)

        detected = score >= threshold
        color = '#d62728' if detected else '#2ca02c'
        status = "이상 탐지" if detected else "정상 판정"
        xlabel = f"score: {score:.4f}  {status}"

        heatmap_rgb = (plt.cm.hot(anomaly_map)[:, :, :3] * 255).astype(np.uint8)

        # 행1: 히트맵 단독
        ax_top = axes[0][i + 1]
        im = ax_top.imshow(heatmap_rgb)
        plt.colorbar(
            plt.cm.ScalarMappable(cmap='hot'),
            ax=ax_top, fraction=0.046, pad=0.04
        )
        ax_top.set_title(cfg["label"], fontsize=11, fontweight='bold')
        ax_top.set_xlabel(xlabel, fontsize=9, color=color, fontweight='bold')
        ax_top.axis('off')

        # 행2: 원본 + 히트맵 오버레이 (원본 65%, 히트맵 35%)
        ax_bot = axes[1][i + 1]
        blended = (original * 0.65 + heatmap_rgb * 0.35).clip(0, 255).astype(np.uint8)
        ax_bot.imshow(blended)
        ax_bot.set_title(f"{cfg['label']} 오버레이", fontsize=10)
        ax_bot.axis('off')

    filename = os.path.basename(img_path)
    fig.suptitle(f"풀링 비교: {filename}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"    저장: {save_path}")


def print_score_table(results):
    labels = [cfg["label"] for cfg in CONFIGS]
    col_w = 14
    header = f"{'이미지':<28}" + "".join(f"{l:>{col_w}}" for l in labels)
    sep = "=" * (28 + col_w * len(CONFIGS))
    print(f"\n{sep}")
    print("  anomaly_score 비교 (높을수록 결함 의심)")
    print(sep)
    print(header)
    print("-" * (28 + col_w * len(CONFIGS)))
    for img_name, scores in results.items():
        row = f"{img_name[:27]:<28}"
        row += "".join(f"{scores[l]:>{col_w}.4f}" for l in labels)
        print(row)
    print(sep)


def compute_auroc_per_config(models_dict, good_paths, bad_paths):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None

    print("\n[INFO] AUROC 계산 중...")
    auroc_results = {}
    for cfg in CONFIGS:
        label = cfg["label"]
        model = models_dict[label]
        labels_list, scores = [], []
        for path in good_paths:
            s, _, _ = run_inference(model, path)
            labels_list.append(0); scores.append(s)
        for path in bad_paths:
            s, _, _ = run_inference(model, path)
            labels_list.append(1); scores.append(s)
        auroc_results[label] = roc_auc_score(labels_list, scores)
    return auroc_results


def save_auroc_bar(auroc_results):
    labels = [cfg["label"] for cfg in CONFIGS]
    vals = [auroc_results[l] for l in labels]
    colors = ['#1f77b4' if v == max(vals) else '#aec7e8' for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=colors, edgecolor='black', width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("풀링 설정", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("풀링 설정별 AUROC 비교", fontsize=13, fontweight='bold')
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
    print("  풀링 방식 비교 실험")
    print(f"  비교 대상: {[c['label'] for c in CONFIGS]}")
    print("=" * 60)

    # ----------------------------------------
    # 1. 모델 준비 (로드 or 학습)
    # ----------------------------------------
    print("\n[STEP 1] 모델 준비")
    models_dict = {}
    for cfg in CONFIGS:
        print(f"\n--- {cfg['label']} ---")
        models_dict[cfg["label"]] = build_model(cfg)

    # ----------------------------------------
    # 2. 결함 이미지 히트맵 비교
    # ----------------------------------------
    bad_paths = collect_images(TEST_BAD_DIR)
    good_paths = collect_images(TEST_GOOD_DIR)

    if not bad_paths:
        print(f"\n[WARN] 결함 이미지 없음: {TEST_BAD_DIR}")
    else:
        print(f"\n[STEP 2] 히트맵 비교 ({len(bad_paths)}장)")
        threshold = 0.5
        score_table = {}
        for img_path in bad_paths:
            fname = os.path.basename(img_path)
            print(f"  처리 중: {fname}")
            scores = {}
            for cfg in CONFIGS:
                s, _, _ = run_inference(models_dict[cfg["label"]], img_path)
                scores[cfg["label"]] = s
            score_table[fname] = scores
            save_path = os.path.join(OUTPUT_DIR, f"compare_{fname}")
            save_comparison_image(img_path, models_dict, threshold, save_path)
        print_score_table(score_table)

    # ----------------------------------------
    # 3. AUROC (good + bad 모두 있을 때)
    # ----------------------------------------
    if good_paths and bad_paths:
        print(f"\n[STEP 3] AUROC 비교 (정상 {len(good_paths)}장 / 결함 {len(bad_paths)}장)")
        auroc_results = compute_auroc_per_config(models_dict, good_paths, bad_paths)
        if auroc_results:
            best = max(auroc_results, key=auroc_results.get)
            for label, auroc in auroc_results.items():
                mark = " ★ 최고" if label == best else ""
                print(f"    {label}: {auroc:.4f}{mark}")
            save_auroc_bar(auroc_results)
    else:
        print("\n[INFO] good 이미지가 있어야 AUROC 계산 가능 (현재 생략)")

    print(f"\n[완료] 결과 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
