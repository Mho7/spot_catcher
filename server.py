"""
FastAPI 백엔드 서버 

실행 방법:
    python server.py
    python server.py --camera 1    <- 액션캠이 1번인 경우

제공 API:
    GET  /health                  - 서버 상태 확인
    GET  /camera/status           - 카메라 목록
    GET  /camera/stream           - 원본 MJPEG 스트리밍
    GET  /camera/stream_detect    - 탐지 MJPEG 스트리밍
    POST /camera/capture          - 1프레임 캡처 + 탐지
    GET  /defects                 - 결함 목록 조회
    GET  /defects/stats           - 결함 통계
"""
import os
import uuid
import time
import numpy as np
from PIL import Image
import cv2
import argparse

from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import uvicorn

from config import IMAGE_SIZE, STATIC_DIR, BASE_DIR, SERVER_HOST, SERVER_PORT
from models.patchcore import PatchCore
from utils.dataset import get_default_transform
from utils.visualization import create_heatmap_overlay, save_single_overlay
from realtime_camera import find_cameras
from database import save_defect, get_defects, get_defect_stats, delete_defect

# ========================================
# 인자 파싱
# ========================================
parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (액션캠은 보통 1)")
args, _ = parser.parse_known_args()

# ========================================
# FastAPI 앱
# ========================================
app = FastAPI(title="표면 결함 탐지 시스템")

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

# ========================================
# 모델 로드 (PatchCore만)
# ========================================
print("[INFO] 서버 시작 중... 모델 로드")

patchcore_model = None
try:
    patchcore_model = PatchCore()
    patchcore_model.load()
    print("[OK] PatchCore 모델 로드 완료")
except Exception as e:
    print(f"[WARN] PatchCore: {e}")

pc_transform = get_default_transform()

# ========================================
# 카메라
# ========================================
camera_cap = None
camera_index = args.camera


def get_camera():
    global camera_cap
    if camera_cap is None or not camera_cap.isOpened():
        camera_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not camera_cap.isOpened():
            camera_cap = cv2.VideoCapture(camera_index)
        if camera_cap.isOpened():
            camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera_cap


# ========================================
# 카메라 API
# ========================================
@app.get("/camera/status")
async def camera_status():
    cameras = find_cameras()
    return {"available_cameras": cameras, "current_index": camera_index}


@app.get("/camera/stream")
async def camera_stream():
    """카메라 원본 MJPEG 스트리밍"""
    def gen():
        cap = get_camera()
        if not cap.isOpened():
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
            time.sleep(0.033)

    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/camera/stream_detect")
async def camera_stream_detect():
    """실시간 탐지 MJPEG 스트리밍 (원본 + 히트맵 나란히)"""
    def gen():
        cap = get_camera()
        if not cap.isOpened():
            return

        if patchcore_model is None or patchcore_model.memory_bank is None:
            return

        count = 0
        last_combined_bgr = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if count % 3 == 0 or last_combined_bgr is None:
                try:
                    pil_img = Image.fromarray(frame_rgb)
                    tensor = pc_transform(pil_img)
                    score, amap = patchcore_model.predict(tensor)

                    resized = np.array(pil_img.resize(IMAGE_SIZE))
                    overlay, _ = create_heatmap_overlay(resized, amap, threshold=0.5, alpha=0.5)

                    combined = np.hstack([resized, overlay])
                    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

                    verdict = "DEFECT" if score > 0.5 else "NORMAL"
                    color = (0, 0, 255) if score > 0.5 else (0, 255, 0)
                    cv2.putText(combined_bgr, f"{verdict} | Score: {score:.4f}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    last_combined_bgr = combined_bgr
                except Exception:
                    if last_combined_bgr is None:
                        last_combined_bgr = frame

            _, buf = cv2.imencode('.jpg', last_combined_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
            time.sleep(0.05)

    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.post("/camera/capture")
async def camera_capture(save_to_db: str = Form("false")):
    """카메라 1프레임 캡처 -> 탐지 -> 결과 반환"""
    try:
        cap = get_camera()
        if not cap.isOpened():
            return JSONResponse(status_code=400, content={"error": f"카메라 {camera_index}번 열 수 없음"})

        ret, frame = cap.read()
        if not ret:
            return JSONResponse(status_code=400, content={"error": "프레임 캡처 실패"})

        if patchcore_model is None or patchcore_model.memory_bank is None:
            return JSONResponse(status_code=400, content={"error": "PatchCore 모델이 준비되지 않았습니다."})

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                    source="camera", model_type="patchcore", anomaly_score=float(score),
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
        "camera_index": camera_index,
        "patchcore_ready": patchcore_model is not None and patchcore_model.memory_bank is not None,
    }


@app.on_event("shutdown")
async def shutdown():
    if camera_cap and camera_cap.isOpened():
        camera_cap.release()


if __name__ == "__main__":
    print(f"\n로컬 서버가 열려요. http://localhost:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
