"""
서버 전역 설정값
데스크톱(GPU 서버)에서 사용하는 경로, 모델 파라미터 등을 관리합니다.
"""
import os

# ============================================
# 경로 설정
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
SAVE_DIR = os.path.join(ROOT_DIR, "saved_models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 저장 폴더 자동 생성
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================
# GLASS 설정
# ============================================
GLASS_BACKBONE = "wideresnet50"
GLASS_LAYERS = ["layer2", "layer3"]
GLASS_PRETRAIN_EMBED_DIM = 1536
GLASS_TARGET_EMBED_DIM = 1536
GLASS_PATCHSIZE = 3
GLASS_DSC_LAYERS = 2
GLASS_DSC_HIDDEN = 1024
GLASS_PRE_PROJ = 1
GLASS_INPUT_SIZE = (1080, 1920)  # (H, W), Colab 학습/체크포인트 설정과 맞출 것

# ============================================
# 탐지 임계값
# ============================================
ANOMALY_THRESHOLD = 0.25
OVERLAY_THRESHOLD = 0.25
# ============================================
# 서버 설정
# ============================================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
