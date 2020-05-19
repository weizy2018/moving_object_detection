import numpy as np 
import cv2 as cv 

capture = cv.VideoCapture('/home/weizy/Programs/opencv/opencv-4.1.0/samples/data/vtest.avi')
ret, frame1 = capture.read()
frame1_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

count = 0
while True:
    ret, frame2 = capture.read()
    if not ret:
        break 
    
    frame2_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    d_frame = cv.absdiff(frame2_gray, frame1_gray)

    ret, d_frame =  cv.threshold(d_frame, 100, 255, cv.THRESH_BINARY)

    cv.imshow("d_frame", d_frame)
    frame1_gray = frame2_gray

    key = cv.waitKey(30)
    if key == ord('q'):
        break
    
    # if key == ord('s'):
    if count == 375:
        cv.imwrite("pic/temporal1_2.jpg", d_frame)
        print("count = ", count)
    
    count = count + 1

capture.release()
cv.destroyAllWindows()