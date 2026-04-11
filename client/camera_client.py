"""
카메라 클라이언트 (노트북에서 실행)

역할:
- 노트북 카메라(또는 액션캠)로 프레임 캡처
- 프레임을 서버(데스크톱)로 전송
- 서버의 탐지 결과를 받아 OpenCV 창에 표시

사용법:
    python camera_client.py
    python camera_client.py --camera 1    # 액션캠이 1번인 경우

조작법:
    q         : 종료
    s         : 현재 결과 이미지 저장
    스페이스바  : 일시정지/재개
"""
import cv2
import time
import argparse
import requests
import numpy as np
from io import BytesIO
from PIL import Image

from config import SERVER_URL, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, \
                   DETECT_INTERVAL, ANOMALY_THRESHOLD, SAVE_TO_DB


def encode_frame(frame_bgr) -> bytes:
    """OpenCV BGR 프레임을 JPEG 바이트로 변환"""
    _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def send_to_server(frame_bytes: bytes) -> dict | None:
    """서버 /detect 엔드포인트에 프레임 전송 후 결과 반환"""
    try:
        resp = requests.post(
            f"{SERVER_URL}/detect",
            files={"file": ("frame.jpg", frame_bytes, "image/jpeg")},
            data={"save_to_db": str(SAVE_TO_DB).lower()},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.ConnectionError:
        print(f"서버 연결 실패: {SERVER_URL}")
    except requests.exceptions.Timeout:
        print("서버 응답 시간 초과")
    return None


def draw_result(frame_bgr, result: dict | None, fps: float) -> np.ndarray:
    """프레임에 탐지 결과 텍스트를 오버레이"""
    display = frame_bgr.copy()

    if result:
        score = result.get("anomaly_score", 0)
        is_anomaly = result.get("is_anomaly", False)
        verdict = result.get("verdict", "")
        infer_time = result.get("inference_time", 0)

        color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        cv2.putText(display, f"{verdict}  Score: {score:.4f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, f"Server infer: {infer_time:.3f}s  |  FPS: {fps:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    else:
        cv2.putText(display, "서버 응답 없음",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(display, f"FPS: {fps:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return display


def main():
    parser = argparse.ArgumentParser(description="카메라 클라이언트")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    args = parser.parse_args()

    # 서버 상태 확인
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=3)
        info = resp.json()
        print(f"서버 연결 성공: {SERVER_URL}")
        print(f"  모델 준비 여부: {info.get('patchcore_ready')}")
        print(f"  디바이스: {info.get('device')}")
    except Exception:
        print(f"경고: 서버({SERVER_URL})에 연결할 수 없습니다. 계속 진행합니다...")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"카메라 {args.camera}번을 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    print(f"카메라 {args.camera}번 시작 ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")

    cv2.namedWindow("Spot Catcher - Client", cv2.WINDOW_NORMAL)

    paused = False
    frame_count = 0
    last_result = None
    fps_start = time.time()
    display_fps = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            # DETECT_INTERVAL 프레임마다 서버로 전송
            if frame_count % DETECT_INTERVAL == 0:
                frame_bytes = encode_frame(frame)
                last_result = send_to_server(frame_bytes)

            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

        display = draw_result(frame, last_result, display_fps)
        cv2.imshow("Spot Catcher - Client", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = f"capture_{int(time.time())}.png"
            cv2.imwrite(save_path, display)
            print(f"저장: {save_path}")
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
