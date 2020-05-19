import numpy as np 
import cv2 as cv 

class Surendra:

    def __init__(self, video_path, max_iterator, alpha=0.005):
        self.video_path = video_path
        self.max_iterator = max_iterator
        self.alpha = alpha

    def getBackground(self):
        # cap = cv.VideoCapture(self.video_path)
        # ret, b0 = cap.read()
        # b0 = cv.cvtColor(b0, cv.COLOR_BGR2GRAY)
        # for i in range(4):
        #     ret, frame = cap.read()
        #     frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #     b0 = b0 + frame
        
        # b0 = b0 / 5
        # b0 = b0.astype(np.uint8)
        # cap.release()

        cap = cv.VideoCapture(self.video_path)
        ret, b0 = cap.read()
        b0 = cv.cvtColor(b0, cv.COLOR_BGR2GRAY)
        b0 = cv.blur(b0, (5, 5))
        for i in range(self.max_iterator):
            ret, frame = cap.read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = cv.blur(frame, (5, 5))
            diff = cv.absdiff(frame, b0)
            # th = self.__getThreshold__(diff)
            # ret, d = cv.threshold(diff, th, 255, cv.THRESH_BINARY)
            ret, d = cv.threshold(diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            b0[d == 255] = self.alpha * frame[d == 255] + (1 - self.alpha) * b0[d == 255]
            b0 = b0.astype(np.uint8)
        
        cap.release()
        return b0
    

    def __getThreshold__(self, img):
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


def sobel_edge(img):
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    grad_x = cv.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad



path = '/home/weizy/Programs/opencv/opencv-4.1.0/samples/data/vtest.avi'
suf = Surendra(path, 100, 0.05)
background = suf.getBackground()
# cv.imwrite("pic/background.jpg", background)
cv.imshow("background", background)
# cv.waitKey(0)

cap = cv.VideoCapture(path)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

frame1_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
frame2_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

frame1_gray = cv.blur(frame1_gray, (5, 5))
frame2_gray = cv.blur(frame2_gray, (5, 5))

edge1 = sobel_edge(frame1_gray)
edge2 = sobel_edge(frame2_gray)

count = 0
while True:
    ret, frame3 = cap.read()
    if not ret:
        break
    frame3_gray = cv.cvtColor(frame3, cv.COLOR_BGR2GRAY)
    frame3_gray = cv.blur(frame3_gray, (5, 5))
    edge3 = sobel_edge(frame3_gray)
    cv.imshow("edge", edge3)

    diff1 = cv.absdiff(edge1, edge2)
    ret, diff1 = cv.threshold(diff1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    diff2 = cv.bitwise_xor(edge2, edge3)
    ret, diff2 = cv.threshold(diff2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    H = cv.bitwise_or(background, diff2)
    dst = cv.bitwise_and(H, diff1)
    
    dst = cv.dilate(dst, (5, 5))

    cv.imshow("dst", dst)

    edge1 = edge2
    edge2 = edge3

    key = cv.waitKey(30)
    if key == ord('q'):
        break
    
    if count == 192:
        cv.imwrite("pic/temporal_diff4.jpg", dst)
        print("OK")
    count += 1


cap.release()
cv.destroyAllWindows()
