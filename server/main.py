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
import uuid
import time
import numpy as np
from PIL import Image
import cv2
import argparse

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

from config import IMAGE_SIZE, STATIC_DIR, BASE_DIR, SERVER_HOST, SERVER_PORT
from models.patchcore import PatchCore
from utils.dataset import get_default_transform
from utils.visualization import create_heatmap_overlay, save_single_overlay
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


@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE_DIR, "..", "client", "mk.html"))


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
        original_np = np.array(pil_image.resize(IMAGE_SIZE))

        tensor = pc_transform(pil_image)
        start = time.time()
        score, anomaly_map = patchcore_model.predict(tensor)
        infer_time = time.time() - start

        rid = str(uuid.uuid4())[:8]
        Image.fromarray(original_np).save(os.path.join(STATIC_DIR, f"cam_{rid}.png"))
        save_single_overlay(original_np, anomaly_map, os.path.join(STATIC_DIR, f"cam_ov_{rid}.png"))

        original_url = f"/static/cam_{rid}.png"
        overlay_url  = f"/static/cam_ov_{rid}.png"

        saved = False
        if save_to_db.lower() == "true":
            try:
                saved = save_defect(
                    source="client_camera", model_type="patchcore", anomaly_score=float(score),
                    original_url=original_url, overlay_url=overlay_url,
                )
            except Exception:
                pass

        return JSONResponse(content={
            "success": True,
            "model": "patchcore",
            "anomaly_score": round(float(score), 4),
            "is_anomaly": score > 0.5,
            "verdict": "결함 탐지" if score > 0.5 else "정상",
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
async def defects_list(limit: int = 100, min_score: float = 0.3):
    data = get_defects(limit=limit, min_score=min_score)
    return {"count": len(data), "defects": data}


@app.delete("/defects/{defect_id}")
async def defect_delete(defect_id: int):
    deleted = delete_defect(defect_id)
    if deleted:
        return {"success": True}
    return JSONResponse(status_code=404, content={"error": "항목을 찾을 수 없어요."})


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
