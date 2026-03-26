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
    """
    
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir: 이미지가 들어있는 폴더 경로
            transform: 이미지 전처리 파이프라인 (None이면 기본값 사용)
        """
        self.image_dir = image_dir
        
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
            print(f"[OK] '{image_dir}'에서 {len(self.image_paths)}개 이미지 로드")
        
        # 전처리 파이프라인
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_default_transform()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """인덱스로 이미지 하나를 가져옴"""
        img_path = self.image_paths[idx]
        
        # 이미지 읽기 (RGB로 변환)
        image = Image.open(img_path).convert('RGB')
        
        # 전처리 적용
        image = self.transform(image)
        
        # 파일명도 함께 반환 (나중에 결과 확인용)
        filename = os.path.basename(img_path)
        
        return image, filename


def get_default_transform():
    """
    기본 이미지 전처리 파이프라인
    
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


def get_dataloader(image_dir, batch_size=8, shuffle=True, transform=None):
    """
    DataLoader 생성 헬퍼 함수
    
    Args:
        image_dir: 이미지 폴더 경로
        batch_size: 한 번에 처리할 이미지 수
        shuffle: 순서 섞기 여부
        transform: 전처리 파이프라인
    """
    dataset = SurfaceDataset(image_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows에서는 0으로 설정 (멀티프로세싱 이슈 방지)
        pin_memory=False  # CPU 모드에서는 False
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
