"""
프로젝트 전역 설정값
여기서 경로, 이미지 크기, 모델 파라미터 등을 관리합니다.
"""
import os

# ============================================
# 경로 설정
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train", "1920")
TEST_GOOD_DIR = os.path.join(DATA_DIR, "test", "good")
TEST_BAD_DIR = os.path.join(DATA_DIR, "test", "bad")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 저장 폴더 자동 생성
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# 정 안되면 224-> 320으로 올려서 미세 결함 잡기
IMAGE_SIZE = (224, 224)  # CPU에서 느리면 128로 줄이기
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================
# PatchCore 설정
# ============================================
PATCHCORE_BACKBONE = "wide_resnet50_2"  # 백본 네트워크
PATCHCORE_LAYERS = ["layer2", "layer3"]  # 특징 추출할 레이어
CORESET_RATIO = 0.1  # 메모리 뱅크에서 유지할 패치 비율 (0.1 = 10%)
# → 값이 작을수록 빠르지만 정확도가 약간 떨어질 수 있음
# → CPU 환경에서는 0.01~0.1 권장

