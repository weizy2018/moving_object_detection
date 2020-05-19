import numpy as np 
import cv2 as cv 

capture = cv.VideoCapture('/home/weizy/Programs/opencv/opencv-4.1.0/samples/data/vtest.avi')
ret, frame1 = capture.read()
ret, frame2 = capture.read()

frame1_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
frame2_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

count = 0
while True:
    ret, frame3 = capture.read()
    if not ret:
        break 
    
    frame3_gray = cv.cvtColor(frame3, cv.COLOR_BGR2GRAY)
    d_frame1 = cv.absdiff(frame2_gray, frame1_gray)
    d_frame2 = cv.absdiff(frame3_gray, frame2_gray)
    
    frame1_gray = frame2_gray
    frame2_gray = frame3_gray

    dst = cv.bitwise_and(d_frame1, d_frame2)

    ret, dst = cv.threshold(dst, 64, 255, cv.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    dst = cv.dilate(dst, kernel)

    cv.imshow("diff", dst)

    key = cv.waitKey(30)
    if key == ord('q'):
        break
    
    # if key == ord('s'):
    if count == 375:
        cv.imwrite("pic/temporal2_2.jpg", dst)
        print("count = ", count)
    
    count = count + 1

capture.release()
cv.destroyAllWindows()