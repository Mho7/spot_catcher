import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.random_projection import SparseRandomProjection
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
       
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = layers
        self.coreset_ratio = coreset_ratio
        
        # 메모리 뱅크 (학습 후 채워짐)
        self.memory_bank = None
  
        
        full_model = getattr(models, backbone)(weights='IMAGENET1K_V1')
        # layer3/layer4/fc 등 미사용 구간을 그래프에서 제외해 VRAM/연산 절감
        return_nodes = {name: name for name in self.layers}
        self.backbone = create_feature_extractor(full_model, return_nodes=return_nodes)
        self.backbone.to(self.device)
        self.backbone.eval()  # 평가 모드 (학습하지 않음)

    
    #  aggregation 방식을 채택함
    def _extract_features(self, images):

        with torch.no_grad():
            feats = self.backbone(images.to(self.device))  # {layer_name: tensor}

        all_features = []
        target_size = None

        for layer_name in self.layers:
            feat = feats[layer_name]

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

        # KNN 추론용 GPU 텐서 캐싱
        self._build_memory_bank_tensor()

    
    def _greedy_coreset(self, features, n_select):

        if features.shape[1] > 128:
            projector = SparseRandomProjection(n_components=128, random_state=42)
            reduced = projector.fit_transform(features)
        else:
            reduced = features
        
        n_total = reduced.shape[0]
        selected_indices = []
        
        # --- [최적화 적용] GPU 가속을 활용해 알고리즘 로직 변경 없이 무식한 연산 속도만 극한으로 올립니다 ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 연산을 위해 GPU로 데이터 이동
        # GPU 메모리(12GB) 내에 충분히 들어가는 크기입니다 (약 1~2GB 내외)
        reduced_tensor = torch.from_numpy(reduced).float().to(device)
        min_distances = torch.full((n_total,), float('inf'), device=device)
        
        first_idx = np.random.randint(n_total)
        selected_indices.append(first_idx)
        
        for i in tqdm(range(1, n_select), desc="   Coreset 선택 (GPU 가속)", leave=False):
            # 가장 최근 선택된 특징점 (Vector)
            last_selected = reduced_tensor[selected_indices[-1]]
            
            # 브로드캐스팅을 통해 전체 행렬과 벡터 간의 유클리디안 거리를 병렬 계산
            distances = torch.norm(reduced_tensor - last_selected, dim=1)
            
            # 기존 최소 거리와 현재 거리 비교 (원래 Numpy의 np.minimum 역할)
            min_distances = torch.minimum(min_distances, distances)
            
            # 가장 멀리 떨어진 특징의 Index 획득
            next_idx = torch.argmax(min_distances).item()
            selected_indices.append(next_idx)
        
        # 메모리 정리 (OOM 방지)
        del reduced_tensor
        del min_distances
        torch.cuda.empty_cache()
        
        return features[selected_indices]
    
    def _build_memory_bank_tensor(self):
        self.memory_bank_tensor = torch.from_numpy(self.memory_bank).float().to(self.device)
        self.memory_bank_sq = (self.memory_bank_tensor ** 2).sum(dim=1, keepdim=True).T  # [1, M]

    def _knn_search(self, patches_tensor):
        # patches_tensor: [N, C] GPU 텐서
        # 메모리 뱅크가 크므로 청크 단위로 나눠 OOM 방지
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

        if self.memory_bank is None:
            raise RuntimeError("먼저 fit()으로 학습을 수행하세요!")

        # 차원 맞추기
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # [3,H,W] → [1,3,H,W]

        # 패치 특징 추출
        patches, (H, W) = self._extract_features(image_tensor)
        patches_tensor = patches.reshape(-1, patches.shape[-1])  # [H*W, C], GPU 유지

        distances = self._knn_search(patches_tensor)  # [H*W], numpy 반환
        anomaly_map = distances.reshape(H, W)

        # 가우시안 스무딩
        anomaly_map = gaussian_filter(anomaly_map, sigma=1)

       
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

        # KNN 추론용 GPU 텐서 재구성
        self._build_memory_bank_tensor()
