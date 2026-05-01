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
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_GOOD_DIR = os.path.join(DATA_DIR, "test", "good")
TEST_BAD_DIR = os.path.join(DATA_DIR, "test", "bad")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 저장 폴더 자동 생성
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# 이미지 설정
IMAGE_SIZE = (720, 1280)  # (Height, Width) 16:9 비율
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================
# SAM 전처리 설정
# ============================================
SAM_MODEL_TYPE = "vit_b"
SAM_MODEL_PATH = os.path.join(SAVE_DIR, "sam_vit_b.pth")
CROP_MARGIN = 20  # 컨투어 크롭 마진 (픽셀)

# 전처리된 이미지 저장 경로
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
PREPROCESSED_TRAIN_DIR = os.path.join(PREPROCESSED_DIR, "train")
PREPROCESSED_TEST_GOOD_DIR = os.path.join(PREPROCESSED_DIR, "test", "good")
PREPROCESSED_TEST_BAD_DIR = os.path.join(PREPROCESSED_DIR, "test", "bad")
os.makedirs(PREPROCESSED_TRAIN_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_TEST_GOOD_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_TEST_BAD_DIR, exist_ok=True)

# ============================================
# PatchCore 설정
# ============================================
PATCHCORE_BACKBONE = "wide_resnet50_2"  # 백본 네트워크
PATCHCORE_LAYERS = ["layer1", "layer2"]  # 특징 추출할 레이어
CORESET_RATIO = 0.1  # 메모리 뱅크에서 유지할 패치 비율