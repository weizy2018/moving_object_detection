import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

# img = cv.imread("pic/rain2.jpg", cv.IMREAD_COLOR)
img = cv.imread("/home/weizy/Programs/YOLOv4/darknet/data/outputframe.jpg", cv.IMREAD_COLOR)

# equ = cv.equalizeHist(img)
# b, g, r = cv.split(img)
# b = cv.equalizeHist(b)
# g = cv.equalizeHist(g)
# r = cv.equalizeHist(r)
# equ = cv.merge((b, g, r))
img[:, :, 0] = cv.equalizeHist(img[:, :, 0])
img[:, :, 1] = cv.equalizeHist(img[:, :, 1])
img[:, :, 2] = cv.equalizeHist(img[:, :, 2])

cv.imshow("img", img)
# cv.imshow("equ", equ)
cv.waitKey(0)
cv.destroyAllWindows()

# plt.subplot(121)
# plt.imshow(equ[:, :, [2, 1, 0]])
# plt.xticks([]), plt.yticks([])

# plt.subplot(122)
# plt.hist(equ.ravel(), 256, [0, 256])
# plt.xticks([]), plt.yticks([])

# plt.show()