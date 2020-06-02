import cv2 as cv 
import numpy as np 

img = cv.imread("pic/frame.jpg", cv.IMREAD_COLOR)

gamma = 4.5
lookUpTable = np.empty((1, 256), np.uint8)

for j in range(256):
    lookUpTable[0, j] = np.clip(pow(j / 255, gamma) * 255, 0, 255)

dst = cv.LUT(img, lookUpTable)

cv.namedWindow("img", cv.WINDOW_NORMAL)
cv.namedWindow("dst", cv.WINDOW_NORMAL)

cv.imshow("img", img)
cv.imshow("dst", dst)
cv.waitKey(0)
cv.destroyAllWindows()