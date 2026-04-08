"""
PatchCore 학습 스크립트

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
import sys
import time
import numpy as np
from PIL import Image

from config import TRAIN_DIR, TEST_GOOD_DIR, TEST_BAD_DIR, STATIC_DIR, IMAGE_SIZE
from models.patchcore import PatchCore
from utils.dataset import get_dataloader, get_default_transform, denormalize
from utils.visualization import save_result_image


def main():
<<<<<<< HEAD
    print("PatchCore 이상 탐지 모델 학습")
=======
>>>>>>> 673f0b54742e8126f992791e667b81dd9982c1f8
    
    # ========================================
    # 1. 데이터 로드
    # ========================================
<<<<<<< HEAD
    print("\n데이터 로드 중...")
    train_loader = get_dataloader(TRAIN_DIR, batch_size=4, shuffle=False)
    
    if len(train_loader.dataset) == 0:
        print("학습 데이터가 없습니다!")
        print(f"   '{TRAIN_DIR}' 폴더에 정상 이미지를 넣어주세요.")
=======
    train_loader = get_dataloader(TRAIN_DIR, batch_size=4, shuffle=False)
    
    if len(train_loader.dataset) == 0:
>>>>>>> 673f0b54742e8126f992791e667b81dd9982c1f8
        return
    
    # ========================================
    # 2. PatchCore 학습 (메모리 뱅크 구축)
    # ========================================
    model = PatchCore()
    
    start_time = time.time()
    model.fit(train_loader)
    train_time = time.time() - start_time
<<<<<<< HEAD
    print(f"\n학습 시간: {train_time:.1f}초")
=======
>>>>>>> 673f0b54742e8126f992791e667b81dd9982c1f8
    
    # ========================================
    # 3. 테스트 이미지로 탐지 테스트
    # ========================================
<<<<<<< HEAD
    print("\n테스트 이미지 탐지 시작")
=======
>>>>>>> 673f0b54742e8126f992791e667b81dd9982c1f8
    
    transform = get_default_transform()
    
    # 테스트 이미지 수집
    test_images = []
    for label, test_dir in [("정상", TEST_GOOD_DIR), ("결함", TEST_BAD_DIR)]:
        if os.path.exists(test_dir):
            for f in sorted(os.listdir(test_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    test_images.append((os.path.join(test_dir, f), label, f))
    
    if not test_images:
<<<<<<< HEAD
        print("테스트 이미지가 없습니다. 학습만 완료되었습니다.")
=======
>>>>>>> 673f0b54742e8126f992791e667b81dd9982c1f8
    else:
        
        results_dir = os.path.join(STATIC_DIR, "patchcore_results")
        os.makedirs(results_dir, exist_ok=True)
        
        for img_path, label, filename in test_images:
            # 이미지 로드 & 전처리
            original_pil = Image.open(img_path).convert('RGB')
            original_np = np.array(original_pil.resize(IMAGE_SIZE))
            
            image_tensor = transform(original_pil)
            
            # 이상 탐지
            start = time.time()
            score, anomaly_map = model.predict(image_tensor)
            infer_time = time.time() - start
            
            # 결과 출력
<<<<<<< HEAD
            status = "이상" if score > 0.5 else "정상"
            print(f"  [{label}] {filename}: 점수={score:.4f} {status} ({infer_time:.2f}초)")
=======
            status = "🔴 이상" if score > 0.5 else "🟢 정상"
>>>>>>> 673f0b54742e8126f992791e667b81dd9982c1f8
            
            # 결과 이미지 저장
            save_path = os.path.join(results_dir, f"result_{filename}")
            save_result_image(original_np, anomaly_map, save_path)
    
    # ========================================
    # 4. 모델 저장
    # ========================================
    model.save()
    
<<<<<<< HEAD
    print("\nPatchCore 학습 및 테스트 완료!")
    print(f"   모델: saved_models/patchcore.pkl")
    print(f"   결과: static/patchcore_results/")
=======
>>>>>>> 673f0b54742e8126f992791e667b81dd9982c1f8


if __name__ == "__main__":
    main()
