import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
def videoCapture():
    cap = cv.VideoCapture("test.mp4")
    if cap.isOpened():
        print("succeed!")
        res, frame = cap.read()
    else:
        print("failed!")
        res = False
    timeF = 40
    c = 1
    while (res):
        res, frame = cap.read()
        if (c % timeF == 0):
            print(frame)
            plt.imshow(frame)
        c = c + 1
        cv.waitKey(1)
    cap.release()

if __name__ == "__main__":
    videoCapture()