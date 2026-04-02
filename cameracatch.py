import cv2
import os

save_dir = "collected data"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽지 x")
        break

    cv2.imshow("Live Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):   # s 누르면 저장
        filename = os.path.join(save_dir, f"frame_{count:04d}.png")
        cv2.imwrite(filename, frame)
        print(f"저장됨: {filename}")
        count += 1

    elif key == ord("q"): # q 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()