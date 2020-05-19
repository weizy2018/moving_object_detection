import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 

MAX_KERNEL_LENGTH = 31

img = cv.imread("pic/blur.jpg", cv.IMREAD_COLOR)

cv.imshow("src", img)

for i in range(MAX_KERNEL_LENGTH):
    print("i = ", i)
    dst = cv.bilateralFilter(img, i, i * 2, i / 2)
    cv.imshow("dst", dst)
    cv.waitKey(0)

cv.destroyAllWindows()