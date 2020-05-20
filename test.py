import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 

# videoPath = "/home/weizy/files/毕设/毕业设计-于孟渤/于孟渤-代码/test.mp4"
videoPath = "/home/weizy/Downloads/my_video2.mp4"
cap = cv.VideoCapture(videoPath)

fps = 30

ret, frame = cap.read()
rs = cv.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
size = rs.shape[:2]
print(size)

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('/home/weizy/Downloads/output.avi',fourcc, fps, (428, 240))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    dst = cv.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
    cv.imshow("src", frame)
    cv.imshow("dst", dst)
    out.write(dst)
    key = cv.waitKey(30)
    if key == ord('q'):
        break
    
cap.release()
out.release()
cv.destroyAllWindows()