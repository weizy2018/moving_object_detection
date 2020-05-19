import cv2 as cv
import numpy as np
import random as rng
rng.seed(12345)

knn_backSub = cv.createBackgroundSubtractorKNN()
mog2_backSub = cv.createBackgroundSubtractorMOG2()
capture = cv.VideoCapture('/home/weizy/Programs/opencv/opencv-4.1.0/samples/data/vtest.avi')
# capture = cv.VideoCapture("/home/weizy/Downloads/custom1.mp4")
# capture = cv.VideoCapture("/home/weizy/files/毕设/毕业设计-于孟渤/于孟渤-代码/test.mp4")
# capture = cv.VideoCapture("/home/weizy/Downloads/testvideo.mp4")
# capture = cv.VideoCapture(0)
if not capture.isOpened:
    print('cannot open the video')
    exit()

cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.namedWindow('mog2', cv.WINDOW_NORMAL)
cv.namedWindow('knn', cv.WINDOW_NORMAL)
cv.namedWindow("canny", cv.WINDOW_NORMAL)
cv.namedWindow("threshold", cv.WINDOW_NORMAL)

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    frame = cv.blur(frame, (5, 5))          # 平滑处理后效果更好
    knn_fgMask = knn_backSub.apply(frame)
    mog2_fgMask = mog2_backSub.apply(frame)

    # ret, dst = cv.threshold(knn_fgMask, 160, 255, cv.THRESH_BINARY)
    ret, dst = cv.threshold(mog2_fgMask, 160, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel)

    # 可以先进行边缘检测后再进行轮廓查找，也可以不进行边缘检测
    edges = cv.Canny(dst, 300, 450)
    # contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv.findContours(mog2_fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    # 删除根据层级关系删除无用轮廓
    index = np.arange(len(contours) - 1, -1, -1)
    for i, c in zip(index, contours[::-1]):
        if hierarchy[i][2] > 0 or hierarchy[i][3] > 0:
            del contours[i]

    for c in contours:
        rect = cv.boundingRect(c)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        x, y, w, h = rect
        # if cv.contourArea(c) > 500:
        if w * h > 400:
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # cv.rectangle(edges, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), 
            (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.imshow('frame', frame)
    cv.imshow('knn', knn_fgMask)
    cv.imshow('mog2', mog2_fgMask)
    # cv.imshow("canny", edges)
    cv.imshow("threshold", dst)
    keyboard = cv.waitKey(30)
    if keyboard == ord('q'):
        break
    if keyboard == ord('s'):
        cv.imwrite("pic/frame.jpg", frame)

capture.release()
cv.destroyAllWindows()


