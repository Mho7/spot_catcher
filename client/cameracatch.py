"""
데이터 수집 스크립트 (노트북에서 실행)

카메라로 정상 이미지를 촬영해 학습 데이터를 모읍니다.
촬영한 이미지는 서버(데스크톱)의 data/train/good/ 으로 옮겨서 학습에 사용하세요.

조작법:
    s: 현재 프레임 저장
    q: 종료
"""
import cv2
import os

from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT

save_dir = "collected_data"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", CAMERA_WIDTH, CAMERA_HEIGHT)

existing = [
    int(f[6:10]) for f in os.listdir(save_dir)
    if f.startswith("frame_") and f.endswith(".png") and f[6:10].isdigit()
]
count = max(existing) + 1 if existing else 0

print(f"카메라 {CAMERA_INDEX}번 시작. 's'로 저장, 'q'로 종료.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        filename = os.path.join(save_dir, f"frame_{count:04d}.png")
        cv2.imwrite(filename, frame)
        print(f"저장: {filename}")
        count += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
