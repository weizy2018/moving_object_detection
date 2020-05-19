import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 

img = cv.imread("pic/rain2.jpg", cv.IMREAD_COLOR)

alpha = [1.5, 2.0, 1.5]
beta = [25, 25, 55]

label = ["(a)", "(b)", "(c)", "(d)"]

plt.subplot(221)
plt.imshow(img[:, :, [2, 1, 0]])
plt.title("原图")
plt.xticks([]), plt.yticks([])
plt.xlabel(label[0])

for a in range(len(alpha)):
    new_img = np.zeros(img.shape, img.dtype)
    print("a = ", a)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                new_img[i, j, k] = np.clip(alpha[a] * img[i, j, k] + beta[a], 0, 255)
    
    plt.subplot(2, 2, a + 2)
    plt.imshow(new_img[:, :, [2, 1, 0]])
    plt.title("alpha = " + str(alpha[a]) + " beta = " + str(beta[a]))
    plt.xticks([]), plt.yticks([])
    plt.xlabel(label[a + 1])

plt.show()

