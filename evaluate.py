"""
PatchCore 모델 평가 스크립트

실행 방법:
    python evaluate.py

이 스크립트가 하는 일:
    1. saved_models/patchcore.pkl 모델 로드
    2. data/test/good/, data/test/bad/ 이미지 전체 추론
    3. AUROC / F1 / Precision / Recall / Accuracy 출력
    4. 최적 임계값 탐색
    5. ROC 커브를 static/roc_curve.png 로 저장
"""
import os
import sys
import numpy as np
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix
)

from config import TEST_GOOD_DIR, TEST_BAD_DIR, SAVE_DIR, STATIC_DIR, IMAGE_SIZE
from models.patchcore import PatchCore
from utils.dataset import get_default_transform


def load_model():
    model = PatchCore()
    model.load()
    print("[OK] 모델 로드 완료")
    return model


def collect_test_images():
    """good/bad 폴더에서 이미지 경로와 라벨(0=정상, 1=이상) 수집"""
    items = []
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for label, folder in [(0, TEST_GOOD_DIR), (1, TEST_BAD_DIR)]:
        if not os.path.exists(folder):
            print(f"[WARN] 폴더 없음: {folder}")
            continue
        files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(valid_ext)]
        for f in files:
            items.append((os.path.join(folder, f), label))

    return items


def run_inference(model, items):
    """모든 이미지에 대해 anomaly_score 추론"""
    transform = get_default_transform()
    scores = []
    labels = []

    print(f"\n추론 시작 (총 {len(items)}장)...")
    for i, (img_path, label) in enumerate(items):
        pil = Image.open(img_path).convert('RGB')
        tensor = transform(pil)
        score, _ = model.predict(tensor)
        scores.append(score)
        labels.append(label)

        tag = "이상" if label == 1 else "정상"
        print(f"  [{i+1:3d}/{len(items)}] {os.path.basename(img_path):30s}  score={score:.4f}  ({tag})")

    return np.array(labels), np.array(scores)


def find_optimal_threshold(labels, scores):
    """ROC 커브에서 Youden's J 기준 최적 임계값 탐색"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    return thresholds[best_idx], fpr, tpr, thresholds


def print_metrics(labels, scores, threshold):
    preds = (scores >= threshold).astype(int)

    auroc = roc_auc_score(labels, scores)
    acc   = accuracy_score(labels, preds)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    f1    = f1_score(labels, preds, zero_division=0)
    cm    = confusion_matrix(labels, preds)

    n_good = int((labels == 0).sum())
    n_bad  = int((labels == 1).sum())

    print("\n" + "=" * 50)
    print("  모델 평가 결과")
    print("=" * 50)
    print(f"  테스트 이미지   : 정상 {n_good}장 / 이상 {n_bad}장")
    print(f"  최적 임계값     : {threshold:.4f}")
    print("-" * 50)
    print(f"  AUROC           : {auroc:.4f}")
    print(f"  Accuracy        : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {rec:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print("-" * 50)
    print("  Confusion Matrix (행=실제, 열=예측)")
    print(f"              예측 정상  예측 이상")
    print(f"  실제 정상  :   {cm[0][0]:5d}      {cm[0][1]:5d}")
    print(f"  실제 이상  :   {cm[1][0]:5d}      {cm[1][1]:5d}")
    print("=" * 50)

    if auroc >= 0.95:
        grade = "우수"
    elif auroc >= 0.85:
        grade = "양호"
    elif auroc >= 0.70:
        grade = "보통"
    else:
        grade = "미흡 (재학습 권장)"
    print(f"  종합 판정       : {grade}  (AUROC {auroc:.3f})")
    print("=" * 50)


def save_roc_curve(fpr, tpr, auroc, save_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import platform

        if platform.system() == 'Darwin':
            plt.rcParams['font.family'] = 'AppleGothic'
        elif platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color='royalblue', lw=2, label=f'ROC (AUROC = {auroc:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.set_title('ROC 커브')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"\n  ROC 커브 저장: {save_path}")
    except Exception as e:
        print(f"\n  [WARN] ROC 커브 저장 실패: {e}")


def main():
    model_path = os.path.join(SAVE_DIR, "patchcore.pkl")
    if not os.path.exists(model_path):
        print(f"[ERROR] 모델 파일이 없습니다: {model_path}")
        print("  먼저 python train_patchcore.py 를 실행하세요.")
        sys.exit(1)

    model = load_model()

    items = collect_test_images()
    if len(items) == 0:
        print("[ERROR] 테스트 이미지가 없습니다.")
        print(f"  '{TEST_GOOD_DIR}' 또는 '{TEST_BAD_DIR}' 에 이미지를 넣어주세요.")
        sys.exit(1)

    n_good = sum(1 for _, l in items if l == 0)
    n_bad  = sum(1 for _, l in items if l == 1)

    if n_good == 0 or n_bad == 0:
        print("[ERROR] AUROC 계산을 위해 정상/이상 이미지가 모두 필요합니다.")
        sys.exit(1)

    labels, scores = run_inference(model, items)

    threshold, fpr, tpr, _ = find_optimal_threshold(labels, scores)

    auroc = roc_auc_score(labels, scores)
    print_metrics(labels, scores, threshold)

    roc_path = os.path.join(STATIC_DIR, "roc_curve.png")
    save_roc_curve(fpr, tpr, auroc, roc_path)


if __name__ == "__main__":
    main()
