import cv2 as cv 
import numpy as np 

def getThreshold(img):
    hist,bins = np.histogram(img.ravel(),256,[0,256])

    img_temp = img.ravel()
    minval = img_temp[0]
    maxval = img_temp[0]

    for i in range(len(img_temp)):
        if minval > img_temp[i]:
            minval = img_temp[i]
        if maxval < img_temp[i]:
            maxval = img_temp[i]

    threshold = 0
    newThreshold = (int)((minval + maxval) / 2)
    while newThreshold != threshold:
        sum1 = sum2 = 0
        w1 = w2 = 0
        for i in range(minval, newThreshold):
            sum1 = sum1 + hist[i] * i
            w1 = w1 + hist[i]
        
        avg1 = sum1 / w1

        for i in range(newThreshold, maxval):
            sum2 = sum2 + hist[i] * i
            w2 = w2 + hist[i]
        
        avg2 = sum2 / w2

        threshold = newThreshold
        newThreshold = (int)((avg1 + avg2) / 2)

    return newThreshold

def robert(img_gray):
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv.filter2D(img_gray, cv.CV_16S, kernelx)
    y = cv.filter2D(img_gray, cv.CV_16S, kernely)

    absX = cv.convertScaleAbs(x)
    absy = cv.convertScaleAbs(y)
    
    dst = cv.addWeighted(absX, 0.5, absy, 0.5, 0)
    return dst


capture = cv.VideoCapture('/home/weizy/Programs/opencv/opencv-4.1.0/samples/data/vtest.avi')
ret, frame1 = capture.read()
ret, frame2 = capture.read()

frame1_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
frame2_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

count = 0
while True:
    ret, frame3 = capture.read()
    if not ret:
        break
    
    frame3_gray = cv.cvtColor(frame3, cv.COLOR_BGR2GRAY)
    d1 = cv.absdiff(frame1_gray, frame2_gray)
    d2 = cv.absdiff(frame2_gray, frame3_gray)
    d3 = cv.absdiff(frame1_gray, frame3_gray)

    t1 = getThreshold(d1)
    t2 = getThreshold(d2)
    t3 = getThreshold(d3)

    ret, d1 = cv.threshold(d1, t1, 255, cv.THRESH_BINARY)
    ret, d2 = cv.threshold(d2, t2, 255, cv.THRESH_BINARY)
    ret, d3 = cv.threshold(d3, t3, 255, cv.THRESH_BINARY)
    # d1 = cv.adaptiveThreshold(d1, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 10)
    # d2 = cv.adaptiveThreshold(d2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 10)
    # d3 = cv.adaptiveThreshold(d3, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 10)

    c1 = cv.bitwise_and(d1, d2)
    c2 = cv.bitwise_and(d2, d3)

    D = cv.bitwise_or(c1, c2)
    s = robert(frame2_gray)

    M = cv.bitwise_and(s, D)
    # 形态学：开运算
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    M = cv.morphologyEx(M, cv.MORPH_OPEN, kernel)

    n = cv.morphologyEx(d3, cv.MORPH_OPEN, kernel)

    dst = cv.bitwise_or(M, n)

    dst = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel)

    # ret, dst = cv.threshold(dst, 10, 255, cv.THRESH_BINARY_INV)

    cv.imshow("dst", dst)
    key = cv.waitKey(30)
    if key == ord('q'):
        break
    
    if key == ord('s'):
        cv.imwrite("pic/temporal_diff3_2.jpg", dst)
        print(count)
    
    if count == 192:
        cv.imwrite("pic/temporal_diff3.jpg", dst)
        print("OK")
    
    count = count + 1

    frame1_gray = frame2_gray
    frame2_gray = frame3_gray

capture.release()
cv.destroyAllWindows()




