import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 

videoPath = "VID.mp4"
cap = cv.VideoCapture(videoPath)

fps = 30

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi',fourcc, fps, (1280, 720))

alpha = 0.5
beta = -10

gamma = 2.0

lookUpTable = np.empty((1, 256), np.uint8)
lookUpTable2 = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(alpha*i + beta, 0, 255)

for i in range(256):
    lookUpTable2[0, i] = np.clip(pow(i / 255, gamma) * 255, 0, 255)

cv.namedWindow("frame", cv.WINDOW_NORMAL)
cv.namedWindow("dst1", cv.WINDOW_NORMAL)
cv.namedWindow("dst2", cv.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    dst1 = cv.LUT(frame, lookUpTable)
    dst2 = cv.LUT(dst1, lookUpTable2)

    cv.imshow("frame", frame)
    cv.imshow("dst1", dst1)
    cv.imshow("dst2", dst2)
    out.write(dst2)
    key = cv.waitKey(30)
    if key == ord('q'):
        break
    
cap.release()
out.release()
cv.destroyAllWindows()