"""
서버 전역 설정값
데스크톱(GPU 서버)에서 사용하는 경로, 모델 파라미터 등을 관리합니다.
"""
import os

# ============================================
# 경로 설정
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 저장 폴더 자동 생성
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ⚠️ [임시] 경량 테스트 모델용 설정 — 동작 확인 / 레이아웃 작업용 (실서비스 모델 아님)
# 실제 모델로 복귀할 땐 아래 "원본" 값으로 되돌릴 것
IMAGE_SIZE = (144, 256)        # 원본: (288, 512)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================
# PatchCore 설정
# ============================================
# ⚠️ [임시] 경량 모델(patchcore_light.pkl) 메타데이터에 맞춘 값
PATCHCORE_BACKBONE = "resnet18"            # 원본: "wide_resnet50_2"
PATCHCORE_LAYERS = ["layer2"]              # 원본: ["layer1", "layer2"]

# ============================================
# 탐지 임계값
# ============================================
ANOMALY_THRESHOLD = 0.7
# ============================================
# 서버 설정
# ============================================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
