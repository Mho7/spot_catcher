import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    if ret:
        print(f"{i} <- 현재 이 카메라가 연동된 상태임")
        cv2.imshow(f"cam {i}", frame)
        cv2.waitKey(0)
        cap.release()
        break