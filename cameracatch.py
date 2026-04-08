import cv2
import os

save_dir = "collected data/1920"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 1920, 1080)

existing = [
    int(f[6:10]) for f in os.listdir(save_dir)
    if f.startswith("frame_") and f.endswith(".png") and f[6:10].isdigit()
]
count = max(existing) + 1 if existing else 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        filename = os.path.join(save_dir, f"frame_{count:04d}.png")
        cv2.imwrite(filename, frame)
        count += 1
        print("저장 완료")
        print(count)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()