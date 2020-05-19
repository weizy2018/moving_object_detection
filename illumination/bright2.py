import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 

img = cv.imread("pic/rain2.jpg", cv.IMREAD_COLOR)

gamma = [0.2, 0.4, 0.6]
lookUpTable = np.empty((1, 256), np.uint8)

plt.subplot(221), plt.title("原图")
plt.imshow(img[:, :, [2, 1, 0]])
plt.xticks([]), plt.yticks([])

for i in range(len(gamma)):
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255, gamma[i]) * 255, 0, 255)
    
    dst = cv.LUT(img, lookUpTable)
    plt.subplot(2, 2, i + 2)
    plt.title("gamma = " + str(gamma[i]))
    plt.imshow(dst[:, :, [2, 1, 0]])
    plt.xticks([]), plt.yticks([])

plt.show()