import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 

img = cv.imread("pic/blur.jpg", cv.IMREAD_COLOR)

# 中值滤波
dst1 = cv.medianBlur(img, 5)

# 双边滤波
dst2 = cv.bilateralFilter(img, 30, 60, 15)

plt.subplot(131)
plt.imshow(img[:, :, [2, 1, 0]])
plt.title("原图")
plt.xticks([]), plt.yticks([])

plt.subplot(132)
plt.imshow(dst1[:, :, [2, 1, 0]])
plt.title("中值滤波")
plt.xticks([]), plt.yticks([])

plt.subplot(133)
plt.imshow(dst2[:, :, [2, 1, 0]])
plt.title("双边滤波")
plt.xticks([]), plt.yticks([])

plt.show()
