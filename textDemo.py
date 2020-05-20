import numpy as np 
import cv2 as cv 

img = cv.imread("pic/blox.jpg")

text_size, baseLine = cv.getTextSize("Hello World", cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
print(text_size)
cv.rectangle(img, (10, 10), (10 + text_size[0], 10 + text_size[1]), (0, 0, 255), cv.FILLED)

cv.putText(img, "Hello World", (10, 10 + text_size[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv.imshow("test", img)
cv.waitKey(0)
cv.destroyAllWindows()