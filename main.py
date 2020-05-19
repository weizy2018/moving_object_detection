import cv2 as cv 
import numpy as np 
import random as rng
import darknet
import os

rng.seed(12345)

videoPath = "/home/weizy/files/毕设/毕业设计-于孟渤/于孟渤-代码/test.mp4"
configPath = "YOLOv4/yolov4.cfg"
weightPath = "YOLOv4/yolov4.weights"
metaPath = "YOLOv4/coco.data"

if not os.path.exists(videoPath):
    raise ValueError("Invalid video path `" + os.path.abspath(videoPath) + "`")
if not os.path.exists(configPath):
    raise ValueError("Invalid config path `" + os.path.abspath(configPath) + "`")
if not os.path.exists(weightPath):
    raise ValueError("Invalid weight path `" + os.path.abspath(weightPath) + "`")
if not os.path.exists(metaPath):
    raise ValueError("Invalid data file path `" + os.path.abspath(metaPath) + "`")

print("test1")
netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
print("test2")
metaMain = darknet.load_meta(metaPath.encode("ascii"))
print("test3")
#从配置文件中找到names文件，并读取coco.names
try:
    with open(metaPath) as metaFH:
        metaContents = metaFH.read()
        import re
        match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
        if match:
            result = match.group(1)
        else:
            result = None
        try:
            if os.path.exists(result):
                with open(result) as namesFH:
                    namesList = namesFH.read().strip().split("\n")
                    altNames = [x.strip() for x in namesList]
        except TypeError:
            pass
except Exception:
    pass
print("test4")
# convert xywh to xyxy
def convert_back(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def detect(img):
    print("test5")
    bgr_img = img[:, :, ::-1]
    height, width = bgr_img.shape[:2]
    rsz_img = cv.resize(bgr_img, (darknet.network_width(netMain), darknet.network_height(netMain)),
                     interpolation=cv.INTER_LINEAR)
    darknet_image, _ = darknet.array_to_image(rsz_img)
    print("test6")
    detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
    print("test7")
    print(len(detections))
    for detection in detections:
        x, y, w, h = detection[2][0], \
                    detection[2][1], \
                    detection[2][2], \
                    detection[2][3]
        conf = detection[1]
        x *= width / darknet.network_width(netMain)
        w *= width / darknet.network_width(netMain)
        y *= height / darknet.network_height(netMain)
        h *= height / darknet.network_height(netMain)
        xyxy = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        label = detection[0].decode()
        index = altNames.index(label)
        label = f'{label} {conf:.2f}'
        print("label: ", label)


# img = cv.imread("pic/low.jpg")
# detect(img)


# sift = cv.xfeatures2d.SIFT_create()
surf = cv.xfeatures2d.SURF_create(400)
FLANN_INDEX_KDTREE = 1
MIN_MATCH_COUNT = 10
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)

cv.namedWindow("match", cv.WINDOW_NORMAL)
cv.namedWindow("frame1", cv.WINDOW_NORMAL)
cv.namedWindow("frame2", cv.WINDOW_NORMAL)
cv.namedWindow("after dilate", cv.WINDOW_NORMAL)
cv.namedWindow("draw contours", cv.WINDOW_NORMAL)
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

        ret2, dst = cv.threshold(sub, 50, 255, cv.THRESH_BINARY)
        # cv.imshow("after threshold", dst)
        dst_gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        dst_gray = cv.morphologyEx(dst_gray, cv.MORPH_DILATE, kernel)
        dst_gray = cv.morphologyEx(dst_gray, cv.MORPH_DILATE, kernel)
        cv.imshow("after dilate", dst_gray)

        # edges = cv.Canny(dst, 300, 450)
        # cv.imshow("edges", edges)
        contours, hierarchy = cv.findContours(dst_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        dst_temp = dst_gray.copy()
        dst_temp = cv.cvtColor(dst_temp, cv.COLOR_GRAY2BGR)
        cv.drawContours(dst_temp, contours, -1, 255, 1)
        cv.imshow("draw contours", dst_temp)
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
                cv.rectangle(temp, (x, y), (x + w, y + h), color, 2)
                # 检测
                # roi = temp[x:x+w, y:y+w]
                roi = temp[y:y+h, x:x+w]
                cv.imshow("ROI", roi)

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
    cv.imshow("frame1", frame1)
    cv.imshow("frame2", frame2)

    key = cv.waitKey(30)
    if key == ord('q'):
        break
    if key == ord('s'):
        cv.imwrite("pic/absdiff.jpg", sub)
    
capture.release()
cv.destroyAllWindows()

