import os
import cv2
import numpy as np
import random
import darknet

netMain = None
metaMain = None
altNames = None

configPath = "YOLOv4/yolov4.cfg"
weightPath = "YOLOv4/yolov4.weights"
metaPath = "YOLOv4/coco.data"
if not os.path.exists(configPath):
    raise ValueError("Invalid config path `" + os.path.abspath(configPath)+"`")
if not os.path.exists(weightPath):
    raise ValueError("Invalid weight path `" + os.path.abspath(weightPath)+"`")
if not os.path.exists(metaPath):
    raise ValueError("Invalid data file path `" + os.path.abspath(metaPath)+"`")

netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
metaMain = darknet.load_meta(metaPath.encode("ascii"))

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

image_name = 'pic/low.jpg'
src_img = cv2.imread(image_name)
bgr_img = src_img[:, :, ::-1]
height, width = bgr_img.shape[:2]
rsz_img = cv2.resize(bgr_img, (darknet.network_width(netMain), darknet.network_height(netMain)),
                     interpolation=cv2.INTER_LINEAR)
darknet_image, _ = darknet.array_to_image(rsz_img)
detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

# convert xywh to xyxy
def convert_back(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


# Plotting functions
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


random.seed(1)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(metaMain.classes)]
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
    plot_one_box(xyxy, src_img, label=label, color=colors[index % metaMain.classes])
# cv2.imwrite('result.jpg', src_img)

cv2.imshow("result", src_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

