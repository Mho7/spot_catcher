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
TRAIN_DIR = os.path.join(DATA_DIR, "train", "good")
TEST_GOOD_DIR = os.path.join(DATA_DIR, "test", "good")
TEST_BAD_DIR = os.path.join(DATA_DIR, "test", "bad")
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 저장 폴더 자동 생성
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# 정 안되면 224-> 320으로 올려서 미세 결함 잡기.. 단 실행 속도가 느려질 수 있음 최후수단
IMAGE_SIZE = (224, 224) 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


PATCHCORE_BACKBONE = "wide_resnet50_2"  # 백본 네트워크
PATCHCORE_LAYERS = ["layer2", "layer3"]
CORESET_RATIO = 0.1  # 메모리 뱅크에서 유지할 패치 비율 -> 이것도 해상도와 같은 느낌 0.1 -> 0.15까지 늘려서 미세결함 잡기

# ============================================
# Aggregation (Pooling) 설정
# ============================================
# USE_AGGREGATION: True = avg_pool2d로 주변 패치 평균 (노이즈 감소, 경계 뭉개짐 위험)
#                 False = 원시 패치 특징 그대로 사용 (노이즈 민감, 미세 결함 검출 유리)
USE_AGGREGATION = False
AGGREGATION_KERNEL_SIZE = 3   # 홀수만 권장: 1(비활성화와 동일), 3, 5, 7


# → 값이 작을수록 빠르지만 정확도가 약간 떨어질 수 있음
# → CPU 환경에서는 0.01~0.1 권장

# ============================================
# 탐지 임계값
# ============================================
ANOMALY_THRESHOLD = 0.3   # 이상 판단 임계값 (0~1, 학습 후 조정 필요)

# ============================================
# 서버 설정
# ============================================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
