import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PATCHCORE_BACKBONE, PATCHCORE_LAYERS, CORESET_RATIO, SAVE_DIR
)


class PatchCore:
  
    
    def __init__(self, backbone=PATCHCORE_BACKBONE, layers=PATCHCORE_LAYERS,
                 coreset_ratio=CORESET_RATIO):
       
        self.device = torch.device("cpu")  # CPU 모드
        self.layers = layers
        self.coreset_ratio = coreset_ratio
        self.memory_bank = None
        self.knn = None

        # ========================================
        # 1. 사전학습된 백본 네트워크 로드
        # ========================================
        
        self.backbone = getattr(models, backbone)(weights='IMAGENET1K_V1')
        self.backbone.to(self.device)
        self.backbone.eval()  # 평가 모드 (학습하지 않음)
        

        self.features = {}
        
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        for layer_name in self.layers:
            layer = dict(self.backbone.named_children())[layer_name]
            layer.register_forward_hook(get_hook(layer_name))
        
    
    #  aggregation 방식을 채택함
    def _extract_features(self, images):

        with torch.no_grad():

            _ = self.backbone(images.to(self.device))
        
        
        all_features = []
        target_size = None
        
        for layer_name in self.layers:
            feat = self.features[layer_name]  
            
            if target_size is None:
                target_size = feat.shape[2:]  # 첫 번째 레이어 크기 기준
            
            # 다른 레이어 크기를 맞춤 (interpolate)
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=False
                )
            
            all_features.append(feat)
        
        
        combined = torch.cat(all_features, dim=1)
        # 추가된 부분 주변 패치 평균으로 
        combined = F.avg_pool2d(combined, kernel_size=3, stride=1, padding=1)

        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
        patches = F.normalize(patches, dim=-1) # 정규화작업 추가함
        return patches, (H, W)
        
    
    def fit(self, dataloader):

        all_patches = []
        

        for images, filenames in tqdm(dataloader, desc="특징 추출"):
            patches, self.feature_map_size = self._extract_features(images)
            all_patches.append(patches.cpu().numpy())
        
     
        all_patches = np.concatenate(all_patches, axis=0)  # [N, num_patches, C]
        all_patches = all_patches.reshape(-1, all_patches.shape[-1])  # [N*num_patches, C]
        
        
  
        n_select = max(1, int(all_patches.shape[0] * self.coreset_ratio))
        
        if n_select < all_patches.shape[0]:
      
            self.memory_bank = self._greedy_coreset(all_patches, n_select)
        else:
            self.memory_bank = all_patches

        # k-NN 탐색 모델 사전 생성 (추론 시 재사용)
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='brute', n_jobs=-1)
        self.knn.fit(self.memory_bank)
        
    
    def _greedy_coreset(self, features, n_select):

        
    
        if features.shape[1] > 128:
            projector = SparseRandomProjection(n_components=128, random_state=42)
            reduced = projector.fit_transform(features)
        else:
            reduced = features
        
        n_total = reduced.shape[0]
        selected_indices = []

        np.random.seed(42)
        first_idx = np.random.randint(n_total)
        selected_indices.append(first_idx)
        

        min_distances = np.full(n_total, np.inf)
        
        for i in tqdm(range(1, n_select), desc="   Coreset 선택", leave=False):
        
            last_selected = reduced[selected_indices[-1]]
            distances = np.sum((reduced - last_selected) ** 2, axis=1)
            min_distances = np.minimum(min_distances, distances)
            
           
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)
        
        return features[selected_indices]
    
    def _knn_search(self, patches):

        patches_sq = np.sum(patches ** 2, axis=1, keepdims=True)
        memory_sq  = np.sum(self.memory_bank ** 2, axis=1, keepdims=True).T
        cross      = patches @ self.memory_bank.T
        dist_sq    = np.maximum(patches_sq + memory_sq - 2 * cross, 0)
        return np.sqrt(dist_sq).min(axis=1)  # [N]

    def predict(self, image_tensor):

        if self.memory_bank is None:
            raise RuntimeError("먼저 fit()으로 학습을 수행하세요!")

        # 차원 맞추기
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # [3,H,W] → [1,3,H,W]

        # 패치 특징 추출
        patches, (H, W) = self._extract_features(image_tensor)
        patches = patches.cpu().numpy().reshape(-1, patches.shape[-1])  # [H*W, C]

        if self.knn is None:
            self.knn = NearestNeighbors(n_neighbors=1, algorithm='brute', n_jobs=-1)
            self.knn.fit(self.memory_bank)

        distances, _ = self.knn.kneighbors(patches)
        distances = distances.flatten()
        anomaly_map = distances.reshape(H, W)

        # 가우시안 스무딩
        anomaly_map = gaussian_filter(anomaly_map, sigma=2)

       
        if anomaly_map.max() > anomaly_map.min():
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())

        top_k = min(5, len(distances))
        anomaly_score = float(np.sort(distances)[-top_k:].mean())

        return anomaly_score, anomaly_map
    
    def save(self, path=None):
        """메모리 뱅크 저장"""
        if path is None:
            path = os.path.join(SAVE_DIR, "patchcore.pkl")
        
        save_data = {
            'memory_bank': self.memory_bank,
            'feature_map_size': self.feature_map_size,
            'layers': self.layers,
            'coreset_ratio': self.coreset_ratio,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, path=None):
        if path is None:
            path = os.path.join(SAVE_DIR, "patchcore.pkl")
        
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.memory_bank = save_data['memory_bank']
        self.feature_map_size = save_data['feature_map_size']

        # k-NN 탐색 모델 즉시 생성
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='brute', n_jobs=-1)
        self.knn.fit(self.memory_bank)
