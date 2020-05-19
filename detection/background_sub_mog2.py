import numpy as np
import cv2 as cv 

capture = cv.VideoCapture('/home/weizy/Programs/opencv/opencv-4.1.0/samples/data/vtest.avi')
# capture = cv.VideoCapture('/home/weizy/Programs/opencv/opencv-4.1.0/samples/data/tree.avi')

# capture = cv.VideoCapture("/home/weizy/Videos/people.mp4")

fgbg = cv.createBackgroundSubtractorMOG2()
count = 0
while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    fgmask = fgbg.apply(frame)
    cv.imshow("frame", fgmask)
    cv.imshow("source", frame)
    key = cv.waitKey(60)
    if key == ord('q'):
        break
    
    if count == 196:
        cv.imwrite("pic/mog2.jpg", fgmask)
        print("save OK")
    
    count = count + 1
    
capture.release()
cv.destroyAllWindows()
