import cv2
import os

save_dir = "collected data"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 1280, 720)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽지 못했습니다.")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        filename = os.path.join(save_dir, f"frame_{count:04d}.png")
        cv2.imwrite(filename, frame)
        print(f"저장됨: {filename}")
        count += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()