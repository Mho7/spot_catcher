# 프로젝트 환경 세팅 가이드

## 프로젝트 구조

```
spot_catcher/
├── requirements.txt
├── SETUP_GUIDE.md
├── client/
│   └── index.html             # 웹 UI (React CDN + Babel standalone)
└── server/                    # FastAPI 백엔드 (GPU 머신에서 실행)
    ├── main.py                # API 엔트리포인트
    ├── config.py              # 경로/모델/서버 설정
    ├── database.py            # 결함 SQLite DB
    ├── models/
    │   └── patchcore.py       # PatchCore 추론 모델
    ├── utils/
    │   ├── dataset.py         # 입력 전처리 transform
    │   └── visualization.py   # 오버레이 생성
    ├── saved_models/
    │   └── patchcore.pkl      # 학습된 memory bank (sik_develop에서 학습 후 복사)
    └── static/                # 런타임 저장 이미지 (save_to_db=true일 때만)
```

학습은 `sik_develop` 브랜치에서 수행하고, 생성된 `patchcore.pkl`을 `server/saved_models/`로 복사해서 이 서버(desktop 브랜치)에서 추론만 돌린다.

## 1단계: Python 설치 (3.10 권장)

```bash
python --version   # Python 3.10.x
```

## 2단계: 가상환경

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # macOS/Linux
```

## 3단계: PyTorch 설치

GPU 환경 (CUDA):
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

## 5단계: 학습된 모델 배치

`sik_develop` 브랜치에서 학습한 `patchcore.pkl`을 다음 경로에 배치:

```
server/saved_models/patchcore.pkl
```

## 6단계: 서버 실행

```bash
cd server
python main.py
```

기본값: `http://0.0.0.0:8000`

브라우저에서 `http://localhost:8000` 접속하면 `client/index.html`이 서빙된다.

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET  | `/health`           | 서버/모델 상태 |
| POST | `/detect`           | 이미지 업로드 → 이상 탐지 결과 |
| GET  | `/defects`          | 저장된 결함 목록 |
| GET  | `/defects/stats`    | 결함 통계 |
| DELETE | `/defects/{id}`   | 결함 삭제 |

`/detect`의 `save_to_db=true`면 이미지를 `server/static/`에 저장하고 URL 반환, `false`면 base64 data URI로 즉시 반환(디스크 누적 방지).

## 자주 발생하는 문제

### "patchcore.pkl을 찾을 수 없습니다"
→ 파일 위치가 `server/saved_models/patchcore.pkl`인지 확인

### "CUDA out of memory"
→ `server/config.py`의 `IMAGE_SIZE`를 줄이거나, PatchCore `_knn_search`의 `chunk_size`를 낮춰볼 것

### "ModuleNotFoundError"
→ `pip install -r requirements.txt` 재실행, 가상환경 활성화 확인
