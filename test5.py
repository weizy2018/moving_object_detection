import cv2 as cv 
import numpy as np 

img = cv.imread("pic/frame.jpg", cv.IMREAD_COLOR)

alpha = 0.4
beta = -10

gamma = 2.0

lookUpTable = np.empty((1, 256), np.uint8)
lookUpTable2 = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(alpha*i + beta, 0, 255)

for i in range(256):
    lookUpTable2[0, i] = np.clip(pow(i / 255, gamma) * 255, 0, 255)

dst1 = cv.LUT(img, lookUpTable)
# dst2 = cv.LUT(dst1, lookUpTable2)

# dst = np.zeros(img.shape, img.dtype)
# for i in range(img.shape[0]):
#     print("i = ", i)
#     for j in range(img.shape[1]):
#         for k in range(img.shape[2]):
#             dst[i, j, k] = np.clip(alpha * img[i, j, k] + beta, 0, 255)

cv.namedWindow("img", cv.WINDOW_NORMAL)
cv.namedWindow("dst1", cv.WINDOW_NORMAL)
# cv.namedWindow("dst2", cv.WINDOW_NORMAL)

# cv.imwrite("/home/weizy/Programs/YOLO/darknet/data/outputframe.jpg", dst2)

cv.imshow("img", img)
cv.imshow("dst1", dst1)
# cv.imshow("dst2", dst2)
cv.waitKey(0)
cv.destroyAllWindows()