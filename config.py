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
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
GLASS_PROJECT_DIR = "/Users/parkhyunsik/파이썬/GLASS"

# 저장 폴더 자동 생성
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# 이미지 정규화 설정
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================
# SAM 전처리 설정
# ============================================
SAM_MODEL_TYPE = "vit_b"
SAM_MODEL_PATH = os.path.join(SAVE_DIR, "sam_vit_b.pth")

# ============================================
# GLASS 추론 데이터 경로
# ============================================
GLASS_CLASSNAME = "spot"
GLASS_DATA_DIR = os.path.join(DATA_DIR, "glass_format")
GLASS_CLASS_DIR = os.path.join(GLASS_DATA_DIR, GLASS_CLASSNAME)
GLASS_TEST_GOOD_DIR = os.path.join(GLASS_CLASS_DIR, "test", "good")
GLASS_TEST_BAD_DIR = os.path.join(GLASS_CLASS_DIR, "test", "bad")

for path in (
    GLASS_TEST_GOOD_DIR,
    GLASS_TEST_BAD_DIR,
):
    os.makedirs(path, exist_ok=True)

# ============================================
# GLASS 모델 설정
# ============================================
GLASS_BACKBONE = "wideresnet50"
GLASS_LAYERS = ["layer2", "layer3"]
GLASS_PRETRAIN_EMBED_DIM = 1536
GLASS_TARGET_EMBED_DIM = 1536
GLASS_PATCHSIZE = 3
GLASS_DSC_LAYERS = 2
GLASS_DSC_HIDDEN = 1024
GLASS_PRE_PROJ = 1
GLASS_INPUT_SIZE = (1080, 1920)  # 추론 시 체크포인트와 맞춘 입력 크기

GLASS_RESULTS_DIR = os.path.join(STATIC_DIR, "glass_results")
os.makedirs(GLASS_RESULTS_DIR, exist_ok=True)
