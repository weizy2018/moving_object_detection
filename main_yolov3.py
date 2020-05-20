import cv2 as cv 
import numpy as np 
import random as rng
import darknet
import os

rng.seed(12345)

videoPath = "test.mp4"
configPath = "YOLOv3/yolov3.cfg"
weightPath = "YOLOv3/yolov3.weights"
classesPath = "YOLOv3/coco.names"

if not os.path.exists(videoPath):
    raise ValueError("Invalid video path `" + os.path.abspath(videoPath) + "`")
if not os.path.exists(configPath):
    raise ValueError("Invalid config path `" + os.path.abspath(configPath) + "`")
if not os.path.exists(weightPath):
    raise ValueError("Invalid weight path `" + os.path.abspath(weightPath) + "`")
if not os.path.exists(classesPath):
    raise ValueError("Invalid classes path `" + os.path.abspath(classesPath) + "`")

with open(classesPath, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

confThreshold = 0.5  # 置信度阈值
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416
inpHeight = 416

net = cv.dnn.readNetFromDarknet(configPath, weightPath)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
# 获取输出层的名字
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 取最大置信度的Bounding Box返回
def process(outs):
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                classIds.append(classId)
                confidences.append(float(confidence))
    
    if len(classIds) != 0:
        index = np.argsort(confidences)[-1]
        classId = classIds[index]
        # print(classes[classId], f'{confidences[index]:.2f}')
        label = f'{classes[classId]} {confidences[index]:.2f}'
        return label
    return None


# surf = cv.xfeatures2d.SURF_create(400)
surf = cv.xfeatures2d.SURF_create()
FLANN_INDEX_KDTREE = 1
MIN_MATCH_COUNT = 10
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 32)
flann = cv.FlannBasedMatcher(index_params, search_params)

# cv.namedWindow("match", cv.WINDOW_NORMAL)
# cv.namedWindow("frame1", cv.WINDOW_NORMAL)
# cv.namedWindow("frame2", cv.WINDOW_NORMAL)
# cv.namedWindow("after dilate", cv.WINDOW_NORMAL)
# cv.namedWindow("draw contours", cv.WINDOW_NORMAL)
cv.namedWindow("rectangle", cv.WINDOW_NORMAL)

# capture = cv.VideoCapture('/home/weizy/Programs/opencv/opencv-4.1.0/samples/data/vtest.avi')
capture = cv.VideoCapture(videoPath)
if not capture.isOpened():
    print("can not open the video")
    exit()

ret, frame2 = capture.read()
rows, cols, ch = frame2.shape
frame2_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
kp2, des2 = surf.detectAndCompute(frame2_gray, None)

RECT_MAX_WIDTH = 0.5 * cols
RECT_MAX_HIGHT = 0.5 * rows
RECT_MIN_WIDTH = 20
RECT_MIN_HIGHT = 20

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
while True:
    frame1 = frame2
    frame1_gray = frame2_gray
    kp1 = kp2
    des1 = des2

    ret, frame2 = capture.read()
    if not ret:
        break
    
    frame2_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # kp1, des1 = surf.detectAndCompute(frame1, None)
    kp2, des2 = surf.detectAndCompute(frame2_gray, None)

    matches = flann.knnMatch(des1, des2, k = 2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        frame1_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        frame2_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(frame1_pts, frame2_pts, cv.RANSAC, 5.0)
        # matches_mask = mask.ravel().tolist()
        
        warp = cv.warpPerspective(frame1, M, (cols, rows))
        sub = cv.absdiff(frame2, warp)
        # cv.imshow("sub", sub)

        # ret2, dst = cv.threshold(sub, 50, 255, cv.THRESH_BINARY)
        # cv.imshow("after threshold", dst)
        # dst_gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
        dst_gray = cv.cvtColor(sub, cv.COLOR_BGR2GRAY)
        ret2, dst_gray = cv.threshold(dst_gray, 50, 255, cv.THRESH_BINARY)

        dst_gray = cv.morphologyEx(dst_gray, cv.MORPH_DILATE, kernel)
        dst_gray = cv.morphologyEx(dst_gray, cv.MORPH_DILATE, kernel)
        # cv.imshow("after dilate", dst_gray)

        # edges = cv.Canny(dst, 300, 450)
        # cv.imshow("edges", edges)
        contours, hierarchy = cv.findContours(dst_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # dst_temp = dst_gray.copy()
        # dst_temp = cv.cvtColor(dst_temp, cv.COLOR_GRAY2BGR)
        # cv.drawContours(dst_temp, contours, -1, 255, 1)
        # cv.imshow("draw contours", dst_temp)
        if hierarchy is None:
            continue
       
        temp = frame2.copy()
        for c in contours:
            rect = cv.boundingRect(c)
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            x, y, w, h = rect
            if cv.contourArea(c) > 200 and w < RECT_MAX_WIDTH and h < RECT_MAX_HIGHT \
                and w > RECT_MIN_WIDTH and h > RECT_MIN_HIGHT:
            # if w * h > 800:
                roi = temp[y:y+h, x:x+w]
                blob = cv.dnn.blobFromImage(roi, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
                net.setInput(blob)
                outputName = getOutputsNames(net)
                outs = net.forward(outputName)
                label = process(outs)

                cv.rectangle(temp, (x, y), (x + w, y + h), color, 2)
                if label:
                    text_size, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv.rectangle(temp, (x, y - text_size[1]), (x + text_size[0], y), color, cv.FILLED)
                    cv.putText(temp, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv.imshow("rectangle", temp)

    else:
        print("not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matches_mask = None
    
    # draw_params = dict(matchColor = (0, 255, 0),
    #                 singlePointColor = None,
    #                 matchesMask = matches_mask,
    #                 flags = 2)
    # img = cv.drawMatches(frame1, kp1, frame2, kp2, good, None, **draw_params)

    # cv.imshow("match", img)
    # cv.imshow("frame1", frame1)
    # cv.imshow("frame2", frame2)

    key = cv.waitKey(20)
    if key == ord('q'):
        break
    if key == ord('s'):
        cv.imwrite("pic/absdiff.jpg", sub)
    
capture.release()
cv.destroyAllWindows()

