"""
데이터 로딩 및 전처리 유틸리티

역할:
- 폴더에서 이미지를 읽어서 PyTorch 텐서로 변환
- ImageNet 정규화 적용 (사전학습 모델용)
- 학습/테스트 데이터 분리
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class SurfaceDataset(Dataset):
    """
    표면 이미지 데이터셋 클래스

    사용법:
        dataset = SurfaceDataset("data/train/good")
        image, filename = dataset[0]  # 첫 번째 이미지와 파일명

        # 데이터 증강 + 반복으로 데이터 수 늘리기
        dataset = SurfaceDataset("data/train/good", augment=True, repeat=20)
    """

    def __init__(self, image_dir, transform=None, augment=False, repeat=1):
        """
        Args:
            image_dir: 이미지가 들어있는 폴더 경로
            transform: 이미지 전처리 파이프라인 (None이면 기본값 사용)
            augment: True이면 학습용 증강 적용
            repeat: 데이터셋 반복 횟수 (이미지 수 * repeat = 실제 학습 샘플 수)
                    augment=True일 때만 의미 있음 (매 반복마다 다른 증강 적용)
        """
        self.image_dir = image_dir
        self.repeat = repeat

        # 지원하는 이미지 확장자
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        # 폴더에서 이미지 파일 목록 가져오기
        self.image_paths = []
        if os.path.exists(image_dir):
            for f in sorted(os.listdir(image_dir)):
                if f.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(image_dir, f))

        if len(self.image_paths) == 0:
            print(f"[WARN] '{image_dir}'에 이미지가 없습니다!")
        else:
            total = len(self.image_paths) * repeat
            print(f"[OK] '{image_dir}'에서 {len(self.image_paths)}개 이미지 로드 "
                  f"(repeat={repeat} → 학습 샘플 {total}개)")

        # 전처리 파이프라인
        if transform is not None:
            self.transform = transform
        elif augment:
            self.transform = get_train_transform()
        else:
            self.transform = get_default_transform()

    def __len__(self):
        return len(self.image_paths) * self.repeat

    def __getitem__(self, idx):
        """인덱스로 이미지 하나를 가져옴 (repeat 적용 시 같은 이미지에 다른 증강)"""
        real_idx = idx % len(self.image_paths)
        img_path = self.image_paths[real_idx]

        # 이미지 읽기 (RGB로 변환)
        image = Image.open(img_path).convert('RGB')

        # 전처리 적용
        image = self.transform(image)

        # 파일명도 함께 반환 (나중에 결과 확인용)
        filename = os.path.basename(img_path)

        return image, filename


def get_train_transform():
    """
    학습용 데이터 증강 파이프라인 (정상 이미지 전용)

    PatchCore는 패치의 공간 위치 정보를 기억하므로
    형태/구조를 왜곡하는 증강(Flip, ResizedCrop 등)은 사용하지 않음.

    적용 증강:
    - RandomRotation(5): 카메라 미세 틀어짐 시뮬레이션 (±5도 이내)
    - RandomAffine(translate): 피사체 미세 위치 변화 (3% 이내)
    - ColorJitter: 공장 조명 밝기/대비 변화 시뮬레이션
    - GaussianBlur: 카메라 초점 흔들림 시뮬레이션
    """
    return transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
        transforms.ColorJitter(brightness=0.3, contrast=0.2),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_default_transform():
    """
    기본 이미지 전처리 파이프라인 (추론/테스트용, 증강 없음)

    1. 리사이즈: 모든 이미지를 동일한 크기로
    2. 텐서 변환: 이미지를 PyTorch 텐서로 (0~1 범위)
    3. 정규화: ImageNet 평균/표준편차로 정규화
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),  # (224, 224)로 리사이즈
        transforms.ToTensor(),          # [0, 255] → [0.0, 1.0]
        transforms.Normalize(           # ImageNet 정규화
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def get_dataloader(image_dir, batch_size=8, shuffle=True, transform=None,
                   augment=False, repeat=1):
    """
    DataLoader 생성 헬퍼 함수

    Args:
        image_dir: 이미지 폴더 경로
        batch_size: 한 번에 처리할 이미지 수
        shuffle: 순서 섞기 여부
        transform: 전처리 파이프라인 (직접 지정 시 augment 무시)
        augment: True이면 학습용 증강 적용
        repeat: 데이터셋 반복 횟수 (augment=True와 함께 사용)
                예) 이미지 9장 + repeat=20 → 180개 학습 샘플
    """
    dataset = SurfaceDataset(image_dir, transform=transform,
                             augment=augment, repeat=repeat)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    return loader


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    정규화된 텐서를 원래 이미지로 되돌리기 (시각화용)
    
    Args:
        tensor: 정규화된 이미지 텐서 [C, H, W]
    Returns:
        numpy 배열 [H, W, C] (0~255 범위, uint8)
    """
    import numpy as np
    
    # 텐서를 numpy로 변환
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # [C,H,W] → [H,W,C]
    
    # 정규화 역변환
    img = img * np.array(std) + np.array(mean)
    
    # 0~255 범위로 클리핑
    img = (img * 255).clip(0, 255).astype(np.uint8)
    
    return img
