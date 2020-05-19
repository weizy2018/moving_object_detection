import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

img = cv.imread("pic/car3.jpg", cv.IMREAD_GRAYSCALE)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
equ = clahe.apply(img)

dst = np.hstack((img, equ))
plt.imshow(dst, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
