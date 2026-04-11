"""
카메라 연결 테스트 (노트북에서 실행)

사용 가능한 카메라 인덱스를 찾습니다.
"""
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    if ret:
        print(f"카메라 {i}번 사용 가능")
        cv2.imshow(f"cam {i}", frame)
        cv2.waitKey(0)
        cap.release()
        break
    cap.release()

cv2.destroyAllWindows()
