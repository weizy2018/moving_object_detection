import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

img = cv.imread("pic/rain2.jpg", cv.IMREAD_COLOR)

plt.subplot(121)
plt.imshow(img[:, :, [2, 1, 0]])
plt.xticks([]), plt.yticks([])

plt.subplot(122)
plt.hist(img.ravel(), 256, [0, 256])
plt.xticks([]), plt.yticks([])

plt.show()

