"""테스트 이미지들로 GLASS 추론 속도 측정"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

import torch
from PIL import Image
from models.glass_detector import GlassDetector

TEST_DIR = r"data\glass_format\spot\test\bad\테스트"

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 모델 로드 시간 측정
    t0 = time.time()
    detector = GlassDetector()
    load_time = time.time() - t0
    print(f"Device: {detector.device}")
    print(f"모델 로드 시간: {load_time:.2f}s\n")

    # 이미지 목록
    images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    images.sort()
    print(f"테스트 이미지: {len(images)}장\n")

    # 워밍업 (첫 추론은 느릴 수 있음)
    warmup_img = Image.open(os.path.join(TEST_DIR, images[0]))
    with torch.no_grad():
        detector.predict(warmup_img)
    print("워밍업 완료\n")

    # 추론 속도 측정
    times = []
    print(f"{'파일명':<20} {'점수':>8} {'시간(ms)':>10}")
    print("-" * 42)

    for fname in images:
        img = Image.open(os.path.join(TEST_DIR, fname))
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.time()
            score, _, _ = detector.predict(img)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.time() - t_start) * 1000

        times.append(elapsed)
        print(f"{fname:<20} {score:>8.4f} {elapsed:>9.1f}ms")

    print("-" * 42)
    avg = sum(times) / len(times)
    print(f"평균: {avg:.1f}ms | 최소: {min(times):.1f}ms | 최대: {max(times):.1f}ms")
    print(f"FPS: {1000/avg:.1f}")

if __name__ == "__main__":
    main()
