# 🚀 프로젝트 환경 세팅 가이드 (Windows + CPU)

## 1단계: Python 설치 (3.10 권장)

1. https://www.python.org/downloads/ 에서 Python 3.10.x 다운로드
2. 설치 시 **"Add Python to PATH"** 반드시 체크!
3. 설치 확인:
```bash
python --version
# Python 3.10.x 가 나오면 성공
```

## 2단계: 프로젝트 폴더 생성 및 가상환경

```bash
# 원하는 위치에 프로젝트 폴더 생성
mkdir anomaly_detection
cd anomaly_detection

# 가상환경 생성 (프로젝트별 패키지 분리)
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 활성화되면 프롬프트 앞에 (venv)가 표시됨
```

## 3단계: PyTorch 설치 (CPU 버전)

⚠️ 이 단계가 가장 중요합니다! GPU가 없으므로 CPU 전용 버전을 설치합니다.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

설치 확인:
```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
# 버전이 출력되고 CUDA: False 이면 정상
```

## 4단계: 나머지 패키지 설치

```bash
pip install -r requirements.txt
```

## 5단계: 프로젝트 폴더 구조

```
anomaly_detection/
├── requirements.txt          # 패키지 목록
├── SETUP_GUIDE.md           # 이 파일
├── config.py                # 설정값 모음
├── models/
│   ├── patchcore.py         # PatchCore 모델
│   └── autoencoder.py       # Autoencoder 모델
├── utils/
│   ├── dataset.py           # 데이터 로딩/전처리
│   └── visualization.py     # 히트맵 시각화
├── train_patchcore.py       # PatchCore 학습 스크립트
├── train_autoencoder.py     # Autoencoder 학습 스크립트
├── evaluate.py              # 모델 비교 평가 스크립트
├── server.py                # FastAPI 서버
├── templates/
│   └── index.html           # 웹 UI
├── static/                  # 정적 파일 (결과 이미지 등)
└── data/
    ├── train/
    │   └── good/            # 정상 이미지만 넣기
    └── test/
        ├── good/            # 테스트용 정상 이미지
        └── bad/             # 테스트용 결함 이미지
```

## 6단계: 데이터 준비

1. `data/train/good/` 폴더에 정상 표면 이미지를 넣으세요 (최소 50장 권장)
2. `data/test/good/` 에 테스트용 정상 이미지 (10~20장)
3. `data/test/bad/` 에 결함 이미지 (10~20장)
4. 이미지 형식: JPG 또는 PNG
5. 가능하면 동일한 촬영 환경(조명, 거리, 각도)을 유지하세요

## 실행 순서

```bash
# 1. PatchCore 학습
python train_patchcore.py

# 2. Autoencoder 학습
python train_autoencoder.py

# 3. 두 모델 성능 비교
python evaluate.py

# 4. 웹 서버 실행
python server.py
# 브라우저에서 http://localhost:8000 접속
```

## 자주 발생하는 문제

### "torch를 찾을 수 없습니다"
→ 가상환경이 활성화되어 있는지 확인 (프롬프트에 (venv) 표시)
→ `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` 재실행

### "CUDA out of memory" 또는 느린 속도
→ CPU 환경에서는 정상입니다. config.py에서 이미지 크기를 줄이면 빨라집니다.

### "ModuleNotFoundError"
→ `pip install -r requirements.txt` 다시 실행

### 이미지가 너무 크다고 나올 때
→ config.py에서 IMAGE_SIZE를 (128, 128)로 줄여보세요
