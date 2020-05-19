import numpy as np 
import cv2 as cv 

img = cv.imread("pic/dark.png", cv.IMREAD_COLOR)

avg = np.mean(img)
print(avg)
k = 0.01
b = 0.31

gamma = k * avg + b

print("gamma = ", gamma)

# gamma = 0.4

lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255, gamma) * 255, 0, 255)
    # lookUpTable[0, i] = np.clip(pow(i, gamma), 0, 255)

dst = cv.LUT(img, lookUpTable)

m = np.median(dst)
c = 2
e = 0.4

dst = 1 / (1 + pow(m / (dst + c), e))

cv.namedWindow("src", cv.WINDOW_NORMAL)
cv.namedWindow("dst", cv.WINDOW_NORMAL)
cv.imshow("src", img)
cv.imshow("dst", dst)
cv.waitKey(0)
cv.destroyAllWindows()