import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from scipy.ndimage import gaussian_filter
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATCHCORE_BACKBONE, PATCHCORE_LAYERS, CORESET_RATIO, SAVE_DIR


class PatchCore:
    def __init__(self, backbone=PATCHCORE_BACKBONE, layers=PATCHCORE_LAYERS, coreset_ratio=CORESET_RATIO):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = layers
        self.coreset_ratio = coreset_ratio
        self.memory_bank = None

        print(f"디바이스: {self.device}")
        print(f"백본 네트워크 로드 중: {backbone}")

        self.backbone = getattr(models, backbone)(weights='IMAGENET1K_V1')
        self.backbone.to(self.device)
        self.backbone.eval()

        self.features = {}

        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook

        for layer_name in self.layers:
            layer = dict(self.backbone.named_children())[layer_name]
            layer.register_forward_hook(get_hook(layer_name))

        print(f"추출 레이어: {self.layers}")
        print("백본 네트워크 준비 완료\n")

    def _extract_features(self, images):
        with torch.no_grad():
            _ = self.backbone(images.to(self.device))

        all_features = []
        target_size = None

        for layer_name in self.layers:
            feat = self.features[layer_name]
            if target_size is None:
                target_size = feat.shape[2:]
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            all_features.append(feat)

        combined = torch.cat(all_features, dim=1)
        combined = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)

        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
        patches = F.normalize(patches, dim=-1)
        return patches, (H, W)

    def predict(self, image_tensor):
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        patches, (H, W) = self._extract_features(image_tensor)
        patches = patches.reshape(-1, patches.shape[-1])  # (N, C) GPU 텐서 유지

        # GPU에서 거리 계산
        dists = torch.cdist(patches, self.memory_bank)  # (N, M)
        distances = dists.min(dim=1).values.cpu().numpy()

        anomaly_map = distances.reshape(H, W)
        anomaly_map = gaussian_filter(anomaly_map, sigma=2)

        if anomaly_map.max() > anomaly_map.min():
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())

        anomaly_score = float(distances.max())

        return anomaly_score, anomaly_map

    def load(self, path=None):
        if path is None:
            path = os.path.join(SAVE_DIR, "patchcore.pkl")

        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        self.memory_bank = torch.tensor(save_data['memory_bank'], dtype=torch.float32).to(self.device)
        self.feature_map_size = save_data['feature_map_size']
