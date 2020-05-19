import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

img = cv.imread("pic/car3.jpg", cv.IMREAD_GRAYSCALE)

equ = cv.equalizeHist(img)

dst = np.hstack((img, equ))

plt.imshow(dst, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.show()




