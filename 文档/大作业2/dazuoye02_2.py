import cv2 as cv
import os
from collections import  deque 
from scipy import linalg
import numpy as np
from matplotlib import pyplot as plt

def MouseCallBack(imgHSV):
    x, y = cv.setMouseCallback("imgHSV", getposHsv)
    return imgHSV[y, x]


def getposHsv(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        return x, y

def genzong(path, Outpath):
    mybuffer = 64
    pts = deque(maxlen=mybuffer)
    cap = cv.VideoCapture(path)
    frame_fps = cap.get(cv.CAP_PROP_FPS)
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter(Outpath, fourcc, frame_fps, (frame_width, frame_height), False)
    (res, frame) = cap.read()
    hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_value = MouseCallBack(hsv_image)
    while res:
        (res, frame) = cap.read()
        if not res: break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, hsv_value, hsv_value)
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
    genzong("目标跟踪/green_ball.mp4", "目标跟踪/green_ball_test.mp4")