import cv2 as cv
import numpy as np

alpha = 0.05

capture = cv.VideoCapture('/home/weizy/Programs/opencv/opencv-4.1.0/samples/data/vtest.avi')
if not capture.isOpened:
    print('cannot open the video')
    exit()

ret, frame = capture.read()
background = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
background = cv.blur(background, (5, 5))

print(background.dtype)

count = 0

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_frame = cv.blur(gray_frame, (5, 5))          # 平滑处理后效果更好

    d_frame = cv.absdiff(gray_frame, background)

    background = alpha * gray_frame + (1 - alpha) * background
    background = background.astype(np.uint8)
    


    cv.imshow("source", frame)
    cv.imshow("diff", d_frame)
    key = cv.waitKey(30)
    if key == ord('q'):
        break
    
    if key == ord('s'):
        cv.imwrite("pic/sub_005.jpg", d_frame)
        print(count)
    if count == 194:
        cv.imwrite("pic/sub_005.jpg", d_frame)
        break
    
    count = count + 1


capture.release()
cv.destroyAllWindows()
    

