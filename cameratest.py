import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    if ret:
        cv2.imshow(f"cam {i}", frame)
        cv2.waitKey(0)
        cap.release()
        break