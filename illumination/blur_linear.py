import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 

img = cv.imread("pic/blur.jpg", cv.IMREAD_COLOR)

# 均值滤波
dst1 = cv.blur(img, (5, 5))

# 高斯滤波
dst2 = cv.GaussianBlur(img, (5, 5), 0)


plt.subplot(131)
plt.imshow(img[:, :, [2, 1, 0]])
plt.title("原图")
plt.xticks([]), plt.yticks([])

plt.subplot(132)
plt.imshow(dst1[:, :, [2, 1, 0]])
plt.title("均值滤波")
plt.xticks([]), plt.yticks([])

plt.subplot(133)
plt.imshow(dst2[:, :, [2, 1, 0]])
plt.title("高斯滤波")
plt.xticks([]), plt.yticks([])

plt.show()