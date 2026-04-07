"""
실시간 카메라 결함 탐지 모듈

역할:
- 액션캠(USB 연결) 영상을 실시간으로 받아서
- PatchCore로 프레임마다 이상 탐지
- 히트맵 오버레이를 실시간으로 생성

사용법 (단독 실행 - 테스트용):
    python realtime_camera.py
    -> OpenCV 창에서 실시간 탐지 결과 확인
    -> 'q' 키로 종료, 's' 키로 현재 프레임 저장
"""
import os
import time
import cv2
import numpy as np
from PIL import Image

from config import IMAGE_SIZE, STATIC_DIR
from utils.dataset import get_default_transform
from utils.visualization import create_heatmap_overlay


class RealtimeDetector:
    """
    실시간 카메라 결함 탐지기

    사용법:
        detector = RealtimeDetector(camera_index=0)
        detector.start()  # OpenCV 창으로 실시간 확인

        # 또는 프레임 단위로 사용 (서버 연동용)
        detector = RealtimeDetector()
        frame = detector.capture_frame()
        result = detector.detect_frame(frame)
    """

    def __init__(self, camera_index=0):
        """
        Args:
            camera_index: 카메라 번호
                - 0: 기본 카메라 (보통 내장 웹캠)
                - 1: 두 번째 카메라 (보통 외장 USB 카메라/액션캠)
                - 2, 3...: 추가 카메라

        액션캠이 인식 안 되면:
            - camera_index를 0, 1, 2 순서로 바꿔보세요
            - 액션캠이 "USB 웹캠 모드"로 설정되어 있는지 확인
        """
        self.camera_index = camera_index
        self.model = None
        self.transform = None
        self.cap = None

        self._load_model()

    def _load_model(self):
        from models.patchcore import PatchCore
        self.model = PatchCore()
        self.model.load()
        self.transform = get_default_transform()
        print("[OK] PatchCore 모델 로드 완료")

    def open_camera(self):
        """
        카메라 연결

        Returns:
            True: 연결 성공
            False: 연결 실패
        """
        print(f"\n[INFO] 카메라 연결 시도 (index: {self.camera_index})...")

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print("   DirectShow 실패, 기본 백엔드로 재시도...")
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"[ERROR] 카메라 {self.camera_index}번을 열 수 없습니다!")
            print("   1. 액션캠이 USB로 연결되어 있는지 확인")
            print("   2. 액션캠을 'USB 웹캠 모드'로 설정")
            print("   3. camera_index를 0, 1, 2로 바꿔서 시도")
            print("   4. 다른 프로그램이 카메라를 사용 중인지 확인")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"[OK] 카메라 연결 성공! 해상도: {w}x{h}, FPS: {fps:.0f}")
        return True

    def close_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("[INFO] 카메라 연결 해제")

    def capture_frame(self):
        """
        카메라에서 프레임 1장 캡처

        Returns:
            numpy array [H, W, 3] (RGB) 또는 None
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def detect_frame(self, frame_rgb):
        """
        단일 프레임에 대해 이상 탐지 수행

        Args:
            frame_rgb: RGB 이미지 numpy array [H, W, 3]

        Returns:
            dict: {
                'anomaly_score': float,
                'is_anomaly': bool,
                'anomaly_map': numpy array,
                'overlay': numpy array (히트맵 오버레이된 이미지),
                'inference_time': float
            }
        """
        pil_image = Image.fromarray(frame_rgb)
        tensor = self.transform(pil_image)

        start = time.time()
        score, anomaly_map = self.model.predict(tensor)
        infer_time = time.time() - start

        resized = np.array(pil_image.resize(IMAGE_SIZE))
        overlay, mask = create_heatmap_overlay(resized, anomaly_map, threshold=0.5)

        is_anomaly = score > 0.5

        return {
            'anomaly_score': float(score),
            'is_anomaly': is_anomaly,
            'anomaly_map': anomaly_map,
            'overlay': overlay,
            'inference_time': infer_time,
        }

    def start(self, show_fps=True):
        """
        OpenCV 창으로 실시간 탐지 시작 (단독 실행 테스트용)

        조작법:
            q: 종료
            s: 현재 프레임 저장
            스페이스바: 일시정지/재개
        """
        if not self.open_camera():
            return

        print("\n" + "=" * 50)
        print("실시간 결함 탐지 시작!")
        print("=" * 50)
        print("   q: 종료")
        print("   s: 현재 프레임 저장")
        print("   스페이스바: 일시정지/재개")
        print("=" * 50)

        paused = False
        frame_count = 0
        fps_start = time.time()
        display_fps = 0

        while True:
            if not paused:
                frame_rgb = self.capture_frame()
                if frame_rgb is None:
                    print("[WARN] 프레임 캡처 실패")
                    continue

                result = self.detect_frame(frame_rgb)
                display = self._create_display(frame_rgb, result, display_fps)

                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    display_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()

            display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            cv2.imshow('Surface Defect Detection', display_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                save_name = f"capture_{int(time.time())}.png"
                save_path = os.path.join(STATIC_DIR, save_name)
                cv2.imwrite(save_path, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
                print(f"[INFO] 프레임 저장: {save_path}")
            elif key == ord(' '):
                paused = not paused
                print("[INFO] 일시정지" if paused else "[INFO] 재개")

        self.close_camera()
        cv2.destroyAllWindows()
        print("\n[INFO] 실시간 탐지 종료")

    def _create_display(self, frame_rgb, result, fps=0):
        """원본(좌) + 히트맵(우) 나란히 + 정보 표시"""
        h, w = IMAGE_SIZE

        original_resized = cv2.resize(frame_rgb, (w, h))
        overlay = result['overlay']
        combined = np.hstack([original_resized, overlay])

        info_bar = np.zeros((60, combined.shape[1], 3), dtype=np.uint8)

        score = result['anomaly_score']
        is_anomaly = result['is_anomaly']
        verdict = "DEFECT DETECTED" if is_anomaly else "NORMAL"
        color = (255, 80, 80) if is_anomaly else (80, 255, 80)

        cv2.putText(info_bar, f"{verdict} | Score: {score:.4f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(info_bar, f"Model: PatchCore | FPS: {fps:.1f} | Infer: {result['inference_time']:.3f}s",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        display = np.vstack([info_bar, combined])
        return display


def find_cameras():
    """사용 가능한 카메라 목록 확인"""
    print("[INFO] 사용 가능한 카메라 검색 중...\n")

    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"   [OK] 카메라 {i}번: {w}x{h}")
            available.append(i)
            cap.release()
        else:
            print(f"   [--] 카메라 {i}번: 없음")

    if not available:
        print("\n[WARN] 연결된 카메라가 없습니다!")
        print("   액션캠을 USB로 연결하고 웹캠 모드로 설정하세요.")
    else:
        print(f"\n[INFO] 액션캠은 보통 {available[-1]}번입니다.")

    return available


# ============================================
# 단독 실행
# ============================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="실시간 결함 탐지")
    parser.add_argument("--camera", type=int, default=0,
                        help="카메라 인덱스 (기본: 0, 액션캠은 보통 1)")
    parser.add_argument("--find", action="store_true",
                        help="연결된 카메라 목록 확인")
    args = parser.parse_args()

    if args.find:
        find_cameras()
    else:
        detector = RealtimeDetector(camera_index=args.camera)
        detector.start()
