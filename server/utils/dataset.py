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
    def __init__(self, image_dir, transform=None, augment=False, repeat=1):
        self.image_dir = image_dir
        self.repeat = repeat

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

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

        if transform is not None:
            self.transform = transform
        elif augment:
            self.transform = get_train_transform()
        else:
            self.transform = get_default_transform()

    def __len__(self):
        return len(self.image_paths) * self.repeat

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_paths)
        img_path = self.image_paths[real_idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        filename = os.path.basename(img_path)
        return image, filename


def get_train_transform():
    return transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
        transforms.ColorJitter(brightness=0.3, contrast=0.2),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_default_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_dataloader(image_dir, batch_size=8, shuffle=True, transform=None,
                   augment=False, repeat=1):
    dataset = SurfaceDataset(image_dir, transform=transform,
                             augment=augment, repeat=repeat)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )
    return loader


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    import numpy as np
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img
