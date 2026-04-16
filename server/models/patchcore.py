import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from scipy.ndimage import gaussian_filter
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATCHCORE_BACKBONE, PATCHCORE_LAYERS, SAVE_DIR


class PatchCore:
    def __init__(self, backbone=PATCHCORE_BACKBONE, layers=PATCHCORE_LAYERS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = layers
        self.memory_bank = None

        full_model = getattr(models, backbone)(weights='IMAGENET1K_V1')
        return_nodes = {layer: layer for layer in self.layers}
        self.backbone = create_feature_extractor(full_model, return_nodes=return_nodes)
        self.backbone.to(self.device)
        self.backbone.eval()

    def _extract_features(self, images):
        with torch.no_grad():
            features = self.backbone(images.to(self.device))

        all_features = []
        target_size = None

        for layer_name in self.layers:
            feat = features[layer_name]
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

    def _build_memory_bank_tensor(self):
        self.memory_bank_tensor = torch.from_numpy(self.memory_bank).float().to(self.device)
        self.memory_bank_sq = (self.memory_bank_tensor ** 2).sum(dim=1, keepdim=True).T  # [1, M]

    def _knn_search(self, patches_tensor):
        chunk_size = 512
        all_distances = []

        with torch.no_grad():
            for i in range(0, patches_tensor.shape[0], chunk_size):
                chunk = patches_tensor[i:i + chunk_size]           # [chunk, C]
                chunk_sq = (chunk ** 2).sum(dim=1, keepdim=True)   # [chunk, 1]
                cross = chunk @ self.memory_bank_tensor.T           # [chunk, M]
                dist_sq = torch.clamp(chunk_sq + self.memory_bank_sq - 2 * cross, min=0)
                distances = dist_sq.sqrt().min(dim=1).values        # [chunk]
                all_distances.append(distances)

        return torch.cat(all_distances).cpu().numpy()  # [N]

    def predict(self, image_tensor):
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        patches, (H, W) = self._extract_features(image_tensor)
        patches_tensor = patches.reshape(-1, patches.shape[-1])  # [H*W, C], GPU 유지

        distances = self._knn_search(patches_tensor)

        anomaly_map = distances.reshape(H, W)
        anomaly_map = gaussian_filter(anomaly_map, sigma=1)

        if anomaly_map.max() > anomaly_map.min():
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())

        top_k = min(5, len(distances))
        anomaly_score = float(np.sort(distances)[-top_k:].mean())

        return anomaly_score, anomaly_map

    def load(self, path=None):
        if path is None:
            path = os.path.join(SAVE_DIR, "patchcore.pkl")

        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        self.memory_bank = save_data['memory_bank']
        self.feature_map_size = save_data['feature_map_size']
        self._build_memory_bank_tensor()
