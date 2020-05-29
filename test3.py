import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

cap = cv.VideoCapture("VID.mp4")
if not cap.isOpened():
    print("can't open the video")
    exit(0)

cv.namedWindow("frame", cv.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv.imshow("frame", frame)
    if (int)(cap.get(cv.CAP_PROP_POS_FRAMES)) == 166:
        cv.imwrite("pic/frame.jpg", frame)
        print("OK")
    
    key = cv.waitKey(30)
    if key == ord('q'):
        break
    

cap.release()
cv.destroyAllWindows()
