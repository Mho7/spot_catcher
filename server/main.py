"""
FastAPI 백엔드 서버 (데스크톱/GPU 서버에서 실행)

실행 방법:
    python main.py

제공 API:
    GET  /health                    - 서버 상태 확인
    POST /detect                    - 이미지 업로드 → 탐지 결과 반환  ← 클라이언트가 주로 사용
    GET  /defects                   - 결함 목록 조회
    GET  /defects/stats             - 결함 통계
    DELETE /defects/{id}            - 결함 삭제
"""
import os
import io
import uuid
import time
import base64
import numpy as np
from PIL import Image
import cv2

from fastapi import FastAPI, Form, File, UploadFile, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

from config import IMAGE_SIZE, STATIC_DIR, BASE_DIR, SERVER_HOST, SERVER_PORT, ANOMALY_THRESHOLD
from models.patchcore import PatchCore
from utils.dataset import get_default_transform
from utils.visualization import make_single_overlay, save_single_overlay
from database import save_defect, get_defects, get_defect_stats, delete_defect

# ========================================
# FastAPI 앱
# ========================================
app = FastAPI(title="Spot Catcher - 표면 결함 탐지 서버")

# 클라이언트(노트북)에서 접근할 수 있도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _to_data_uri(image_np: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image_np).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE_DIR, "..", "client", "index.html"))


# ========================================
# 모델 로드
# ========================================
patchcore_model = None
try:
    patchcore_model = PatchCore()
    patchcore_model.load()
    print("PatchCore 모델 로드 완료")
except Exception as e:
    print(f"PatchCore 모델 로드 실패: {e}")

pc_transform = get_default_transform()


# ========================================
# 탐지 API  ← 클라이언트(노트북)가 프레임을 올려서 결과를 받아가는 핵심 엔드포인트
# ========================================
@app.post("/detect")
async def detect(file: UploadFile = File(...), save_to_db: str = Form("false")):
    """
    클라이언트(노트북 카메라)에서 캡처한 이미지를 받아 이상 탐지 수행

    Request:
        file      : 이미지 파일 (JPEG/PNG)
        save_to_db: "true"면 결함 DB에 저장

    Response:
        anomaly_score : 이상 점수 (0~1)
        is_anomaly    : 결함 여부
        verdict       : "결함 탐지" or "정상"
        inference_time: 추론 소요 시간(초)
        original_url  : 원본 이미지 URL  (/static/...)
        overlay_url   : 오버레이 이미지 URL
    """
    try:
        if patchcore_model is None or patchcore_model.memory_bank is None:
            return JSONResponse(status_code=503, content={"error": "모델이 아직 준비되지 않았습니다."})

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return JSONResponse(status_code=400, content={"error": "이미지 디코딩 실패"})

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        original_np = np.array(pil_image.resize(IMAGE_SIZE[::-1]))  # PIL은 (W,H), IMAGE_SIZE는 (H,W)

        tensor = pc_transform(pil_image)
        start = time.time()
        score, anomaly_map = patchcore_model.predict(tensor)
        infer_time = time.time() - start

        should_save = save_to_db.lower() == "true"

        if should_save:
            rid = str(uuid.uuid4())[:8]
            Image.fromarray(original_np).save(os.path.join(STATIC_DIR, f"cam_{rid}.png"))
            save_single_overlay(original_np, anomaly_map, os.path.join(STATIC_DIR, f"cam_ov_{rid}.png"))
            original_url = f"/static/cam_{rid}.png"
            overlay_url = f"/static/cam_ov_{rid}.png"
        else:
            overlay_np = make_single_overlay(original_np, anomaly_map)
            original_url = _to_data_uri(original_np)
            overlay_url = _to_data_uri(overlay_np)

        saved = False
        if should_save:
            try:
                saved = save_defect(
                    source="client_camera", model_type="patchcore", anomaly_score=float(score),
                    original_url=original_url, overlay_url=overlay_url,
                    inference_time=infer_time,
                )
            except Exception as e:
                print(f"DB 저장 실패: {e}")

        return JSONResponse(content={
            "success": True,
            "model": "patchcore",
            "anomaly_score": round(float(score), 4),
            "is_anomaly": score > ANOMALY_THRESHOLD,
            "verdict": "결함 탐지" if score > ANOMALY_THRESHOLD else "정상",
            "inference_time": round(infer_time, 3),
            "original_url": original_url,
            "overlay_url": overlay_url,
            "saved_to_db": saved,
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ========================================
# 결함 DB 조회 API
# ========================================
@app.get("/defects")
async def defects_list(
    limit: int = 100,
    min_score: float = 0.3,
    max_score: float = None,
    start_date: str = None,
    end_date: str = None,
    min_inference_time: float = None,
    max_inference_time: float = None,
):
    """
    검색/필터 query params:
        min_score / max_score              : 결함율 범위 (0~1)
        start_date / end_date              : 날짜 범위 "YYYY-MM-DD"
        min_inference_time / max_inference_time : 추론 시간 범위 (초)
    """
    data = get_defects(
        limit=limit,
        min_score=min_score, max_score=max_score,
        start_date=start_date, end_date=end_date,
        min_inference_time=min_inference_time,
        max_inference_time=max_inference_time,
    )
    return {"count": len(data), "defects": data}


@app.delete("/defects/{defect_id}")
async def defect_delete(defect_id: int):
    # row 삭제 전에 파일 경로 미리 조회 후 같이 unlink
    import sqlite3
    from database import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT original_url, overlay_url FROM defects WHERE id = ?", (defect_id,)
    ).fetchone()
    conn.close()

    if row is None:
        return JSONResponse(status_code=404, content={"error": "항목을 찾을 수 없어요."})

    delete_defect(defect_id)

    for url in (row["original_url"], row["overlay_url"]):
        if url and url.startswith("/static/"):
            fpath = os.path.join(STATIC_DIR, url[len("/static/"):])
            try:
                os.remove(fpath)
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"파일 삭제 실패 ({fpath}): {e}")

    return {"success": True}


def _save_data_uri(data_uri: str, prefix: str) -> str:
    """data URI를 disk에 저장하고 /static/ 경로 반환."""
    _, b64 = data_uri.split(",", 1)
    img_bytes = base64.b64decode(b64)
    rid = str(uuid.uuid4())[:8]
    filename = f"{prefix}_{rid}.png"
    with open(os.path.join(STATIC_DIR, filename), "wb") as f:
        f.write(img_bytes)
    return f"/static/{filename}"


@app.post("/defects/save_current")
async def defects_save_current(payload: dict = Body(...)):
    """이미 detect된 결과를 사후 DB에 저장 (선택 버튼). 임계값 무시."""
    score = float(payload.get("anomaly_score", 0))
    inference_time = payload.get("inference_time")
    original_url = payload.get("original_url") or ""
    overlay_url = payload.get("overlay_url") or ""

    # data URI면 disk에 떨어뜨리고 정적 경로로 교체
    if original_url.startswith("data:"):
        original_url = _save_data_uri(original_url, "cam")
    if overlay_url.startswith("data:"):
        overlay_url = _save_data_uri(overlay_url, "cam_ov")

    saved = save_defect(
        source="client_camera", model_type="patchcore",
        anomaly_score=score,
        original_url=original_url, overlay_url=overlay_url,
        inference_time=inference_time,
        force=True,
    )
    return {"saved": bool(saved), "original_url": original_url, "overlay_url": overlay_url}


@app.get("/defects/stats")
async def defects_stats():
    return get_defect_stats()


@app.get("/health")
async def health():
    return {
        "status": "running",
        "device": str(patchcore_model.device) if patchcore_model else "N/A",
        "patchcore_ready": patchcore_model is not None and patchcore_model.memory_bank is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
