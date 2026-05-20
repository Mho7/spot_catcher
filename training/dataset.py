"""GLASS 재학습용 PyTorch 데이터셋"""
import os
import sys
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from vendor.glass.perlin import generate_thr

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _transform(h, w):
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class NormalImageDataset(Dataset):
    """학습용: 정상 이미지 + Perlin 합성 이상 생성."""
    distribution = 3  # hypersphere — 분포 판단 단계 건너뜀

    def __init__(self, image_paths, input_size=(1080, 1920)):
        self.paths = list(image_paths)
        h, w = input_size
        self.feat_h, self.feat_w = h // 8, w // 8
        self.tf = _transform(h, w)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_t = self.tf(Image.open(self.paths[idx]).convert("RGB"))
        aug_t, mask_s = self._synthesize(img_t)
        return {"image": img_t, "aug": aug_t, "mask_s": mask_s}

    def _synthesize(self, img_t):
        C, H, W = img_t.shape
        mask_img = torch.from_numpy(generate_thr((C, H, W), min=0, max=4)).float()
        noise = torch.empty_like(img_t).uniform_(-2.0, 2.0)
        alpha = random.uniform(0.3, 0.7)
        mask_3d = mask_img.unsqueeze(0).expand_as(img_t)
        aug_t = img_t * (1 - mask_3d) + (img_t * (1 - alpha) + noise * alpha) * mask_3d
        mask_s = F.max_pool2d(
            mask_img.unsqueeze(0).unsqueeze(0),
            kernel_size=(H // self.feat_h, W // self.feat_w),
        ).squeeze().float()
        return aug_t, mask_s


class EvalDataset(Dataset):
    """검증용: 정상/결함 이미지 + 레이블. mask_gt 없음 → pixel_auroc=-1(무시)."""
    def __init__(self, normal_paths, defect_paths, input_size=(1080, 1920)):
        h, w = input_size
        self.tf = _transform(h, w)
        self.items = [(str(p), 0) for p in normal_paths] + [(str(p), 1) for p in defect_paths]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        return {"image": self.tf(Image.open(path).convert("RGB")),
                "is_anomaly": label, "image_path": path}
