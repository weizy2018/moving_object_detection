import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

img = cv.imread("pic/rain2.jpg", cv.IMREAD_COLOR)

# equ = cv.equalizeHist(img)
b, g, r = cv.split(img)
b = cv.equalizeHist(b)
g = cv.equalizeHist(g)
r = cv.equalizeHist(r)
equ = cv.merge((b, g, r))

plt.subplot(121)
plt.imshow(equ[:, :, [2, 1, 0]])
plt.xticks([]), plt.yticks([])

plt.subplot(122)
plt.hist(equ.ravel(), 256, [0, 256])
plt.xticks([]), plt.yticks([])

plt.show()