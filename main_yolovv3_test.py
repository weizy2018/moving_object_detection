import cv2 as cv 
import numpy as np 
import random as rng
import darknet
import os

rng.seed(12345)

# videoPath = "VID.mp4"
videoPath = "output.avi"
configPath = "YOLOv3/yolov3.cfg"
weightPath = "YOLOv3/yolov3.weights"
# configPath = "YOLOv3/yolov3-tiny.cfg"
# weightPath = "YOLOv3/yolov3-tiny.weights"
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

confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 320
inpHeight = 320

net = cv.dnn.readNetFromDarknet(configPath, weightPath)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
# 获取输出层的名字
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 取最大置信度的Bounding Box返回
# 该函数不用了
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

def postProcess(roi, outs):
    height, width, channel = roi.shape
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = (int)(detection[0] * width)
                center_y = (int)(detection[1] * height)
                box_width = (int)(detection[2] * width)
                box_height = (int)(detection[3] * height)
                left = (int)(center_x - box_width / 2)
                top = (int)(center_y - box_height / 2)
                classIds.append(classId)
                confidences.append((float)(confidence))
                boxes.append([left, top, box_width, box_height])
    box_result = []
    conf_result = []
    class_result = []
    index = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in index:
        i = i[0]
        box_result.append(boxes[i])
        conf_result.append(confidences[i])
        class_result.append(classes[classIds[i]])
    
    # print(class_result)
    return box_result, conf_result, class_result

def transform(box, roi_x, roi_y):
    for b in box:
        b[0] += roi_x
        b[1] += roi_y


surf = cv.xfeatures2d.SURF_create(1000)
# surf = cv.xfeatures2d.SURF_create()
FLANN_INDEX_KDTREE = 1
MIN_MATCH_COUNT = 10
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 32)
flann = cv.FlannBasedMatcher(index_params, search_params)

# cv.namedWindow("match", cv.WINDOW_NORMAL)
# cv.namedWindow("frame1", cv.WINDOW_NORMAL)
# cv.namedWindow("frame2", cv.WINDOW_NORMAL)
cv.namedWindow("after dilate", cv.WINDOW_NORMAL)
cv.namedWindow("threshold", cv.WINDOW_NORMAL)
cv.namedWindow("sub", cv.WINDOW_NORMAL)
# cv.namedWindow("draw contours", cv.WINDOW_NORMAL)
cv.namedWindow("dst", cv.WINDOW_NORMAL)

capture = cv.VideoCapture(videoPath)
if not capture.isOpened():
    print("can not open the video")
    exit()

alpha = 4
beta = 20
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(alpha*i + beta, 0, 255)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
ret, frame2 = capture.read()
rows, cols, ch = frame2.shape
print("rows = ", rows, " cols = ", cols)
frame_copy = frame2.copy()
frame_copy = cv.LUT(frame_copy, lookUpTable)

frame2_gray = cv.cvtColor(frame_copy, cv.COLOR_BGR2GRAY)
frame2_gray = cv.blur(frame2_gray, (5, 5))
kp2, des2 = surf.detectAndCompute(frame2_gray, None)

RECT_MAX_WIDTH = 0.5 * cols
RECT_MAX_HIGHT = 0.5 * rows
RECT_MIN_WIDTH = 20
RECT_MIN_HIGHT = 20

# test = True
test = False

while True:
    frame1 = frame2
    frame1_gray = frame2_gray
    kp1 = kp2
    des1 = des2

    ret, frame2 = capture.read()
    if not ret:
        break
    frame_copy = frame2.copy()
    frame_copy = cv.LUT(frame_copy, lookUpTable)
    # cv.imshow("frame_copy", frame_copy)
    frame2_gray = cv.cvtColor(frame_copy, cv.COLOR_BGR2GRAY)
    frame2_gray = cv.blur(frame2_gray, (5, 5))

    kp2, des2 = surf.detectAndCompute(frame2_gray, None)
    matches = flann.knnMatch(des1, des2, k = 2)
    # print(len(matches))
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        frame1_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        frame2_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(frame1_pts, frame2_pts, cv.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        
        warp = cv.warpPerspective(frame1, M, (cols, rows))
        sub = cv.absdiff(frame2, warp)
        cv.imshow("sub", sub)
        # if (int)(capture.get(cv.CAP_PROP_POS_FRAMES)) == 166:
        #     cv.imwrite("pic/sub.jpg", sub)
        #     print("OK")

        sub = sub[20:rows - 20, 20:cols - 20]

        # con_img = sub.copy()

        # 通过实验测试，先进行阈值处理然后再转换成灰度图的效果
        # 比先转换成灰度图后再进行阈值处理的效果好
        # 50
        ret2, dst = cv.threshold(sub, 8, 255, cv.THRESH_BINARY)
        dst_gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
        # dst_gray = cv.cvtColor(sub, cv.COLOR_BGR2GRAY)
        # ret2, dst_gray = cv.threshold(dst_gray, 80, 255, cv.THRESH_BINARY)

        cv.imshow("threshold", dst_gray)

        dst_gray = cv.morphologyEx(dst_gray, cv.MORPH_DILATE, kernel)
        dst_gray = cv.morphologyEx(dst_gray, cv.MORPH_DILATE, kernel)
        cv.imshow("after dilate", dst_gray)

        # edges = cv.Canny(dst, 300, 450)
        # cv.imshow("edges", edges)
        contours, hierarchy = cv.findContours(dst_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(con_img, contours, -1, 255, 1)
        # cv.imshow("draw contours", con_img)
        if hierarchy is None:
            continue
       
        temp = frame2[20:rows - 20, 20:cols - 20].copy()
        for c in contours:
            rect = cv.boundingRect(c)
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            x, y, w, h = rect
            # if cv.contourArea(c) > 200 and w < RECT_MAX_WIDTH and h < RECT_MAX_HIGHT \
            #     and w > RECT_MIN_WIDTH and h > RECT_MIN_HIGHT and w * h > 5000:
            if cv.contourArea(c) > 200 and w > RECT_MIN_WIDTH and h > RECT_MIN_HIGHT and w * h > 5000:
                label = None
                if not test:
                    roi = temp[y:y+h, x:x+w]
                    blob = cv.dnn.blobFromImage(roi, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
                    net.setInput(blob)
                    outputName = getOutputsNames(net)
                    outs = net.forward(outputName)
                    # label = process(outs)
                    box, conf, cla = postProcess(roi, outs)
                    transform(box, x, y)
                    t, _ = net.getPerfProfile()
                    # print("Inference time: %.2f ms" % (t * 1000.0 / cv.getTickFrequency()))

                if len(box) > 0:
                    for i, b in enumerate(box):
                        label = f'{cla[i]} {conf[i]:.2f}'
                        tex_size, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv.rectangle(temp, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), color, 2)
                        cv.rectangle(temp, (b[0], b[1] - tex_size[1]), (b[0] + tex_size[0], b[1]), color, cv.FILLED)
                        cv.putText(temp, label, (b[0], b[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                else:
                    cv.rectangle(temp, (x, y), (x + w, y + h), color, 2)
                # if label and not test:
                #     text_size, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                #     cv.rectangle(temp, (x, y - text_size[1]), (x + text_size[0], y), color, cv.FILLED)
                #     cv.putText(temp, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        totalFrame = (int)(capture.get(cv.CAP_PROP_FRAME_COUNT))
        frameNum = (int)(capture.get(cv.CAP_PROP_POS_FRAMES))
        if (int)(capture.get(cv.CAP_PROP_POS_FRAMES)) == 166:
            cv.imwrite("pic/rect_low.jpg", temp)
            print("OK")
        text = "current frame: " + str(frameNum) + "/" + str(totalFrame)
        cv.putText(temp, text, (5, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv.imshow("dst", temp)

    else:
        print("not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matches_mask = None
    
    # draw_params = dict(matchColor = (0, 255, 0),
    #                 singlePointColor = None,
    #                 matchesMask = matches_mask,
    #                 flags = 2)
    # match = cv.drawMatches(frame1, kp1, frame2, kp2, good, None, **draw_params)

    # cv.imshow("match", match)

    key = cv.waitKey(20)
    if key == ord('q'):
        break
    if key == ord('s'):
        cv.imwrite("pic/absdiff.jpg", sub)
    if key == ord('m'):
        cv.imwrite("pic/match.jpg", match)
    if key == ord('t'):
        cv.imwrite("pic/threshold.jpg", dst_gray)
    
capture.release()
cv.destroyAllWindows()

