import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 

img = cv.imread("pic/lena.jpg", cv.IMREAD_COLOR)
plt.imshow(img[:, :, ::-1])
plt.show()