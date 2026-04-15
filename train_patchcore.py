"""
PatchCore 학습 스크립트

실행 방법:
    python train_patchcore.py

이 스크립트가 하는 일:
    1. data/train/good/ 에서 정상 이미지 로드
    2. PatchCore 메모리 뱅크 구축 (특징 추출 + Coreset)
    3. 모델을 saved_models/patchcore.pkl 로 저장
"""
import os

from config import TRAIN_DIR
from models.patchcore import PatchCore
from utils.dataset import get_dataloader


def main():

    # ========================================
    # 1. 데이터 로드
    # ========================================
    print("\n데이터 로드 중...")
    train_loader = get_dataloader(TRAIN_DIR, batch_size=4, shuffle=True,
                                  augment=True, repeat=5)

    if len(train_loader.dataset) == 0:
        print("학습 데이터가 없습니다!")
        print(f"   '{TRAIN_DIR}' 폴더에 정상 이미지를 넣어주세요.")
        return

    # ========================================
    # 2. PatchCore 학습 (메모리 뱅크 구축)
    # ========================================
    model = PatchCore()

    model.fit(train_loader)

    # ========================================
    # 3. 모델 저장
    # ========================================
    model.save()
    


if __name__ == "__main__":
    main()
