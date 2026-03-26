"""
PatchCore 모델 구현

원리 요약:
1. 사전학습된 WideResNet50에서 중간 레이어의 특징(feature)을 패치 단위로 추출
2. 정상 이미지들의 패치 특징을 메모리 뱅크에 저장
3. Coreset Subsampling으로 핵심 특징만 선별 → 속도 향상
4. 새 이미지 입력 시, 패치 특징과 메모리 뱅크 간 거리 비교로 이상 판단

참고 논문: "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022)
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
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
    """
    PatchCore 이상 탐지 모델
    
    사용법:
        model = PatchCore()
        model.fit(train_dataloader)           # 정상 이미지로 메모리 뱅크 구축
        score, anomaly_map = model.predict(image_tensor)  # 이상 탐지
        model.save("saved_models/patchcore.pkl")
        model.load("saved_models/patchcore.pkl")
    """
    
    def __init__(self, backbone=PATCHCORE_BACKBONE, layers=PATCHCORE_LAYERS,
                 coreset_ratio=CORESET_RATIO):
        """
        Args:
            backbone: 사전학습 백본 네트워크 이름
            layers: 특징을 추출할 레이어 이름 리스트
            coreset_ratio: 메모리 뱅크에서 유지할 특징 비율
        """
        self.device = torch.device("cpu")  # CPU 모드
        self.layers = layers
        self.coreset_ratio = coreset_ratio
        
        # 메모리 뱅크 (학습 후 채워짐)
        self.memory_bank = None
        
        # ========================================
        # 1. 사전학습된 백본 네트워크 로드
        # ========================================
        print(f"📦 백본 네트워크 로드 중: {backbone}")
        print("   (최초 실행 시 모델 다운로드에 시간이 걸릴 수 있습니다)")
        
        self.backbone = getattr(models, backbone)(weights='IMAGENET1K_V1')
        self.backbone.to(self.device)
        self.backbone.eval()  # 평가 모드 (학습하지 않음)
        
        # ========================================
        # 2. 중간 레이어 특징 추출을 위한 Hook 등록
        # ========================================
        # Hook: 네트워크의 특정 레이어에서 출력값을 가로채는 기능
        self.features = {}
        
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        for layer_name in self.layers:
            layer = dict(self.backbone.named_children())[layer_name]
            layer.register_forward_hook(get_hook(layer_name))
        
        print(f"   추출 레이어: {self.layers}")
        print("✅ 백본 네트워크 준비 완료\n")
    
    def _extract_features(self, images):
        """
        이미지 배치에서 패치 단위 특징 추출
        
        Args:
            images: 이미지 텐서 [B, 3, H, W]
        Returns:
            패치 특징 [B, num_patches, feature_dim]
        """
        with torch.no_grad():
            # 백본에 이미지 통과 → Hook으로 중간 특징 수집
            _ = self.backbone(images.to(self.device))
        
        # 여러 레이어의 특징을 결합
        all_features = []
        target_size = None
        
        for layer_name in self.layers:
            feat = self.features[layer_name]  # [B, C, H_feat, W_feat]
            
            if target_size is None:
                target_size = feat.shape[2:]  # 첫 번째 레이어 크기 기준
            
            # 다른 레이어 크기를 맞춤 (interpolate)
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=False
                )
            
            all_features.append(feat)
        
        # 채널 방향으로 결합: [B, C1+C2, H_feat, W_feat]
        combined = torch.cat(all_features, dim=1)
        
        # [B, C, H, W] → [B, H*W, C] (각 패치의 특징 벡터로 변환)
        B, C, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        return patches, (H, W)
    
    def fit(self, dataloader):
        """
        정상 이미지들로 메모리 뱅크 구축 (학습 단계)
        
        Args:
            dataloader: 정상 이미지 DataLoader
        """
        print("=" * 50)
        print("🔧 PatchCore 학습 시작 (메모리 뱅크 구축)")
        print("=" * 50)
        
        all_patches = []
        
        # ========================================
        # Step 1: 모든 정상 이미지에서 패치 특징 추출
        # ========================================
        print("\n📸 정상 이미지에서 패치 특징 추출 중...")
        for images, filenames in tqdm(dataloader, desc="특징 추출"):
            patches, self.feature_map_size = self._extract_features(images)
            all_patches.append(patches.cpu().numpy())
        
        # [전체 이미지 × 패치 수, 특징 차원] 형태로 결합
        all_patches = np.concatenate(all_patches, axis=0)  # [N, num_patches, C]
        all_patches = all_patches.reshape(-1, all_patches.shape[-1])  # [N*num_patches, C]
        
        print(f"   총 패치 수: {all_patches.shape[0]:,}")
        print(f"   패치 특징 차원: {all_patches.shape[1]}")
        
        # ========================================
        # Step 2: Coreset Subsampling (핵심 패치만 선별)
        # ========================================
        # 모든 패치를 저장하면 메모리/속도 문제 → 대표적인 것만 남김
        n_select = max(1, int(all_patches.shape[0] * self.coreset_ratio))
        print(f"\n🎯 Coreset Subsampling: {all_patches.shape[0]:,} → {n_select:,} 패치")
        
        if n_select < all_patches.shape[0]:
            # Greedy Coreset: 가장 멀리 떨어진 패치를 순차적으로 선택
            self.memory_bank = self._greedy_coreset(all_patches, n_select)
        else:
            self.memory_bank = all_patches
        
        print(f"   메모리 뱅크 크기: {self.memory_bank.shape}")
        print("\n✅ PatchCore 학습 완료!")
    
    def _greedy_coreset(self, features, n_select):
        """
        Greedy Coreset Selection
        
        원리: 이미 선택된 점들과 가장 먼 점을 반복적으로 선택
        → 전체 분포를 잘 대표하는 부분집합을 선택하는 효과
        
        Args:
            features: 전체 패치 특징 [N, C]
            n_select: 선택할 패치 수
        Returns:
            선택된 패치 특징 [n_select, C]
        """
        print("   Coreset 선택 중 (시간이 걸릴 수 있습니다)...")
        
        # 차원 축소로 속도 향상 (옵션)
        if features.shape[1] > 128:
            projector = SparseRandomProjection(n_components=128, random_state=42)
            reduced = projector.fit_transform(features)
        else:
            reduced = features
        
        n_total = reduced.shape[0]
        selected_indices = []
        
        # 첫 번째 점: 랜덤 선택
        first_idx = np.random.randint(n_total)
        selected_indices.append(first_idx)
        
        # 각 점에서 가장 가까운 선택된 점까지의 거리
        min_distances = np.full(n_total, np.inf)
        
        for i in tqdm(range(1, n_select), desc="   Coreset 선택", leave=False):
            # 마지막으로 선택된 점과의 거리 계산
            last_selected = reduced[selected_indices[-1]]
            distances = np.linalg.norm(reduced - last_selected, axis=1)
            
            # 최소 거리 업데이트
            min_distances = np.minimum(min_distances, distances)
            
            # 가장 먼 점 선택
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)
        
        return features[selected_indices]
    
    def predict(self, image_tensor):
        """
        단일 이미지의 이상 탐지
        
        Args:
            image_tensor: 전처리된 이미지 텐서 [1, 3, H, W] 또는 [3, H, W]
        
        Returns:
            anomaly_score: 이미지 전체 이상 점수 (float, 높을수록 이상)
            anomaly_map: 픽셀별 이상 점수 맵 (numpy, [H_orig, W_orig])
        """
        if self.memory_bank is None:
            raise RuntimeError("먼저 fit()으로 학습을 수행하세요!")
        
        # 차원 맞추기
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # [3,H,W] → [1,3,H,W]
        
        # 패치 특징 추출
        patches, (H, W) = self._extract_features(image_tensor)
        patches = patches.cpu().numpy().reshape(-1, patches.shape[-1])  # [H*W, C]
        
        # ========================================
        # 메모리 뱅크와의 거리 계산
        # ========================================
        # 각 패치에 대해 메모리 뱅크에서 가장 가까운 정상 패치와의 거리
        distances = []
        for patch in patches:
            # L2 거리 계산
            dist = np.linalg.norm(self.memory_bank - patch, axis=1)
            min_dist = dist.min()  # 가장 가까운 정상 패치와의 거리
            distances.append(min_dist)
        
        distances = np.array(distances)
        
        # 이상 맵: [H*W] → [H, W]로 reshape
        anomaly_map = distances.reshape(H, W)
        
        # 가우시안 스무딩 (부드러운 히트맵을 위해)
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        
        # 0~1 범위로 정규화
        if anomaly_map.max() > anomaly_map.min():
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        
        # 전체 이상 점수 = 최대 패치 거리
        anomaly_score = float(distances.max())
        
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
        print(f"💾 PatchCore 모델 저장: {path}")
    
    def load(self, path=None):
        """메모리 뱅크 로드"""
        if path is None:
            path = os.path.join(SAVE_DIR, "patchcore.pkl")
        
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.memory_bank = save_data['memory_bank']
        self.feature_map_size = save_data['feature_map_size']
        print(f"📂 PatchCore 모델 로드: {path}")
        print(f"   메모리 뱅크 크기: {self.memory_bank.shape}")
