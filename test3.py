import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 

img = cv.imread("pic/threshold.jpg", cv.IMREAD_COLOR)
plt.imshow(img[:, :, ::-1])
plt.show()