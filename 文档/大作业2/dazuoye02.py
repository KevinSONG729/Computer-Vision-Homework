import cv2 as cv
import os
from scipy import linalg
import numpy as np
from matplotlib import pyplot as plt

def runing_detect(path):
    cap = cv.VideoCapture(path)
    frame_fps = cap.get(cv.CAP_PROP_FPS)
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    kernel_1 = np.ones((3, 3), np.uint8)
    kernel_2 = np.ones((2, 2), np.uint8)
    print(frame_fps)
    print(frame_height)
    print(frame_width)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter("test3.mp4", fourcc, frame_fps, (frame_width, frame_height), False)
    if cap.isOpened():
        print("succeed!")
        res, frame = cap.read()
        first = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    else:
        print("failed!")
        res = False
    while (res):
        res, frame = cap.read()
        if not res: break

        sub = cv.absdiff(first, cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        # first = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        _, sub = cv.threshold(sub, 244, 255, cv.THRESH_BINARY|cv.THRESH_TRIANGLE)
        sub = cv.erode(sub, kernel_1, iterations=1)
        sub = cv.dilate(sub, kernel_2, iterations=2)
        sub = cv.blur(sub,(4,4))
        writer.write(sub)
    cap.release()
    writer.release()

def MOG_KNN(path):
    cap = cv.VideoCapture(path)
    frame_fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    bg_detect = cv.createBackgroundSubtractorMOG2(history=int(frame_count), varThreshold=150, detectShadows=True)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter("test31.mp4", fourcc, frame_fps, (frame_width, frame_height), False)
    if cap.isOpened():
        print("succeed!")
        res, frame = cap.read()
    else:
        print("failed!")
        res = False
    while (res):
        res, frame = cap.read()
        if not res: break
        mask = bg_detect.apply(cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        writer.write(mask)
    cap.release()
    writer.release()

def generate_video(path1, path2, path3):
    cap1 = cv.VideoCapture(path1)
    cap2 = cv.VideoCapture(path2)
    frame_width = int(cap1.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cap1.get(cv.CAP_PROP_FPS)
    frame_count = cap1.get(cv.CAP_PROP_FRAME_COUNT)
    writer = cv.VideoWriter(path3, cv.VideoWriter_fourcc(*'mp4v'), frame_fps, (frame_width, frame_height // 2))
    res1,res2 = True,True
    while res1 and res2:
        res1, frame1 = cap1.read()
        res2, frame2 = cap2.read()
        if not (res1 and res2): break
        frame1 = cv.resize(frame1, (frame_width // 2, frame_height // 2))
        frame2 = cv.resize(frame2, (frame_width // 2, frame_height // 2))
        img = np.hstack((frame1, frame2))
        writer.write(img)
    writer.release()
    cap1.release()
    cap2.release()


if __name__ == "__main__":
    print("dazuoye02")
    # runing_detect("运动检测视频/station.avi")
    # MOG_KNN("运动检测视频/station.avi")
    generate_video("运动检测视频/school.avi","test2.mp4","test2_.mp4")