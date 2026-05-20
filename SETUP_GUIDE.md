# 프로젝트 환경 세팅 가이드

## 프로젝트 구조

```text
spot_catcher/
├── requirements.txt
├── SETUP_GUIDE.md
├── client/
│   └── index.html             # 웹 UI
├── saved_models/
│   └── ckpt_best_*.pth        # Colab/GLASS에서 학습한 체크포인트
└── server/
    ├── main.py                # FastAPI API 엔트리포인트
    ├── config.py              # GLASS/경로/서버 설정
    ├── database.py            # 결함 SQLite DB
    ├── models/
    │   └── glass_detector.py  # GLASS 추론 래퍼
    ├── vendor/
    │   └── glass/             # GLASS 원본 코드 (추론용, 자가완결)
    ├── utils/
    │   └── visualization.py   # 히트맵 생성
    └── static/                # 결함 이미지 저장 위치
```

학습은 Colab/별도 환경에서 수행하고, 생성된 `ckpt_best_*.pth`를 `saved_models/`에 배치해서 이 서버에서 추론만 실행한다. GLASS 모델 코드는 `server/vendor/glass/`에 포함되어 외부 경로 의존이 없다.

## 1단계: Python 설치

Python 3.10 계열을 권장한다.

```bash
python --version
```

## 2단계: 가상환경

```bash
python -m venv venv
venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
```

## 3단계: PyTorch 설치

GPU 환경:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU 전용:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

확인:

```bash
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

## 4단계: 의존성 설치

```bash
pip install -r requirements.txt
```

## 5단계: 체크포인트 준비

Colab/GLASS에서 학습한 체크포인트 한 개를 다음 위치에 둔다.

```text
saved_models/ckpt_best_*.pth
```

## 6단계: 서버 실행

```bash
cd server
python main.py
```

브라우저에서 `http://localhost:8000`에 접속하면 `client/index.html`이 서빙된다.

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 서버/GLASS 모델 상태 |
| POST | `/detect` | 이미지 업로드 → GLASS 이상 탐지 결과 |
| GET | `/defects` | 저장된 결함 목록 |
| GET | `/defects/stats` | 결함 통계 |
| DELETE | `/defects/{id}` | 결함 삭제 |

`/detect`는 카메라 이미지를 업로드받아 GLASS score/map을 계산한다. 결함으로 판정되면 이미지를 `server/static/`에 저장하고 DB에 기록하며, 정상이면 base64 data URI로 즉시 반환한다.

## 자주 발생하는 문제

### "GLASS 체크포인트를 찾지 못했습니다"

`saved_models/ckpt_best_*.pth`가 있는지 확인한다.

### "CUDA out of memory"

`server/config.py`의 `GLASS_INPUT_SIZE`를 Colab 학습 크기와 맞춰 낮춰본다.

### "ModuleNotFoundError"

가상환경 활성화 후 `pip install -r requirements.txt`를 다시 실행한다.
