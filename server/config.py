"""
서버 전역 설정값
데스크톱(GPU 서버)에서 사용하는 경로, 모델 파라미터 등을 관리합니다.
"""
import os

# ============================================
# 경로 설정
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train", "good")
TEST_GOOD_DIR = os.path.join(DATA_DIR, "test", "good")
TEST_BAD_DIR = os.path.join(DATA_DIR, "test", "bad")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 저장 폴더 자동 생성
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

IMAGE_SIZE = (320, 320)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================
# PatchCore 설정
# ============================================
PATCHCORE_BACKBONE = "wide_resnet50_2"
PATCHCORE_LAYERS = ["layer2", "layer3"]
CORESET_RATIO = 0.1

# ============================================
# 탐지 임계값
# ============================================
ANOMALY_THRESHOLD = 0.3

# ============================================
# 서버 설정
# ============================================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
