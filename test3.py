import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

cap = cv.VideoCapture("output.avi")
if not cap.isOpened():
    print("can't open the video")
    exit(0)

cv.namedWindow("frame", cv.WINDOW_NORMAL)

alpha = 4
beta = 20
gamma = 0.6

lookUpTable = np.empty((1, 256), np.uint8)
lookUpTable2 = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(alpha*i + beta, 0, 255)

for i in range(256):
    lookUpTable2[0, i] = np.clip(pow(i / 255, gamma) * 255, 0, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame_copy = frame.copy()
    # frame_copy[:, :, 0] = cv.equalizeHist(frame_copy[:, :, 0])
    # frame_copy[:, :, 1] = cv.equalizeHist(frame_copy[:, :, 1])
    # frame_copy[:, :, 2] = cv.equalizeHist(frame_copy[:, :, 2])
    dst1 = cv.LUT(frame, lookUpTable)
    dst2 = cv.LUT(dst1, lookUpTable2)

    cv.imshow("frame", frame)
    cv.imshow("dst1", dst1)
    cv.imshow("dst2", dst2)

    # if (int)(cap.get(cv.CAP_PROP_POS_FRAMES)) == 166:
    #     cv.imwrite("pic/frame.jpg", frame)
    #     print("OK")
    
    key = cv.waitKey(30)
    if key == ord('q'):
        break
    

cap.release()
cv.destroyAllWindows()
