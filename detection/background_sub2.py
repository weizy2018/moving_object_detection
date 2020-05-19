import cv2 as cv
import numpy as np

capture = cv.VideoCapture('/home/weizy/Programs/opencv/opencv-4.1.0/samples/data/vtest.avi')
if not capture.isOpened:
    print('cannot open the video')
    exit()

ret, frame = capture.read()
background = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
background = cv.blur(background, (5, 5))

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_frame = cv.blur(gray_frame, (5, 5))          # 平滑处理后效果更好
    d_frame = cv.absdiff(gray_frame, background)



    cv.imshow("source", frame)
    cv.imshow("diff", d_frame)
    key = cv.waitKey(30)
    if key == ord('q'):
        break
    
    if key == ord('s'):
        cv.imwrite("pic/sub1.jpg", d_frame)

capture.release()
cv.destroyAllWindows()
    

