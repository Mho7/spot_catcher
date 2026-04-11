"""
PatchCore 학습 스크립트 (서버/데스크톱에서 실행)

실행 방법:
    python train_patchcore.py

이 스크립트가 하는 일:
    1. data/train/good/ 에서 정상 이미지 로드
    2. PatchCore 메모리 뱅크 구축 (특징 추출 + Coreset)
    3. data/test/ 에서 테스트 이미지로 탐지 테스트
    4. 결과 히트맵을 static/ 폴더에 저장
    5. 모델을 saved_models/patchcore.pkl 로 저장
"""
import os
import time
import numpy as np
from PIL import Image

from config import TRAIN_DIR, TEST_GOOD_DIR, TEST_BAD_DIR, STATIC_DIR, IMAGE_SIZE
from models.patchcore import PatchCore
from utils.dataset import get_dataloader, get_default_transform
from utils.visualization import save_result_image


def main():

    # ========================================
    # 1. 데이터 로드
    # ========================================
    print("\n데이터 로드 중...")
    train_loader = get_dataloader(TRAIN_DIR, batch_size=4, shuffle=True,
                                  augment=True, repeat=20)

    if len(train_loader.dataset) == 0:
        print("학습 데이터가 없습니다!")
        print(f"   '{TRAIN_DIR}' 폴더에 정상 이미지를 넣어주세요.")
        return

    # ========================================
    # 2. PatchCore 학습 (메모리 뱅크 구축)
    # ========================================
    model = PatchCore()

    start_time = time.time()
    model.fit(train_loader)
    train_time = time.time() - start_time
    print(f"학습 시간: {train_time:.1f}초")

    # ========================================
    # 3. 테스트 이미지로 탐지 테스트
    # ========================================
    transform = get_default_transform()

    test_images = []
    for label, test_dir in [("정상", TEST_GOOD_DIR), ("결함", TEST_BAD_DIR)]:
        if os.path.exists(test_dir):
            for f in sorted(os.listdir(test_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    test_images.append((os.path.join(test_dir, f), label, f))

    if not test_images:
        print("테스트 이미지가 없습니다.")
    else:
        results_dir = os.path.join(STATIC_DIR, "patchcore_results")
        os.makedirs(results_dir, exist_ok=True)

        for img_path, label, filename in test_images:
            original_pil = Image.open(img_path).convert('RGB')
            original_np = np.array(original_pil.resize(IMAGE_SIZE))
            image_tensor = transform(original_pil)

            start = time.time()
            score, anomaly_map = model.predict(image_tensor)
            infer_time = time.time() - start

            status = "이상" if score > 0.5 else "정상"
            print(f"[{label}] {filename}: {status} (score={score:.4f}, {infer_time:.3f}s)")

            save_path = os.path.join(results_dir, f"result_{filename}")
            save_result_image(original_np, anomaly_map, save_path)

    # ========================================
    # 4. 모델 저장
    # ========================================
    model.save()


if __name__ == "__main__":
    main()
