import cv2 as cv
import os
from collections import  deque 
from scipy import linalg
import numpy as np
from matplotlib import pyplot as plt

def getposHsv(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param[1].append(param[0][y, x])

def MouseCallBack(imgHSV):
    cv.imshow("imgHSV", imgHSV)
    hsv_value_list = []
    cv.setMouseCallback("imgHSV", getposHsv, [imgHSV, hsv_value_list])
    cv.waitKey(10000)
    h_list = []
    s_list = []
    v_list = []
    for i in hsv_value_list:
        h_list.append(int(i[0]))
        s_list.append(int(i[1]))
        v_list.append(int(i[2]))
    H_l, H_h, S_l, S_h, V_L, V_h = min(h_list), max(h_list), min(s_list), max(s_list), min(v_list), max(v_list)
    colorL = (H_l, S_l, V_L)
    colorH = (H_h, S_h, V_h)
    print(colorL, colorH)
    return colorL, colorH

def genzong(path, Outpath):
    mybuffer = 16
    pts = deque(maxlen=mybuffer)
    cap = cv.VideoCapture(path)
    frame_fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter(Outpath, fourcc, frame_fps, (frame_width, frame_height), True)
    detect = cv.createBackgroundSubtractorKNN(history=20, detectShadows=True)
    i = 0
    while(i<100):
        (res, frame) = cap.read()
        i = i + 1
    hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    colorL, colorH = MouseCallBack(hsv_image)
    while res:
        (res, frame) = cap.read()
        if not res: break
        #
        # frame_knn = detect.apply(frame)
        # frame_knn = cv.cvtColor(frame_knn, cv.COLOR_GRAY2BGR)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # mask = cv.inRange(hsv, colorL, colorH)
        mask = cv.inRange(hsv, (0,20,5), (50,31,33))
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        if len(cnts) > 0:
            c = max(cnts, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(c)
            M = cv.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius > 0:
                cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv.circle(frame, center, 5, (0, 0, 255), -1)
                pts.appendleft(center)
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(mybuffer / float(i + 1)) * 2.5)
            cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        writer.write(frame)
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    print("dazuoye02_2")
    genzong("目标跟踪/rat.mp4", "目标跟踪/rat_test.mp4")