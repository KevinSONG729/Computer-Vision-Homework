import cv2 as cv
import os
from scipy import linalg
import numpy as np
from matplotlib import pyplot as plt

def HarrFaceDetect(path, savename, arg1, arg2):
    img_row = cv.imread(path, 1)
    img = cv.imread(path, 0)
    face_detector = cv.CascadeClassifier("E:/haarcascades/haarcascade_frontalface_alt2.xml")
    faces = face_detector.detectMultiScale(img, arg1, arg2)
    for i in faces:
        cv.rectangle(img_row, (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), (0, 255, 0), 2)
    cv.imshow("img", img_row)
    cv.imwrite(savename, img_row)
    cv.waitKey()

if __name__ == '__main__':
    print("face detect")
    # HarrFaceDetect("2-a.jpg", "2-a-detect.jpg", 1.04, 5)
    # HarrFaceDetect("2-b.jpg", "2-b-detect.jpg", 1.03, 5)
    # HarrFaceDetect("myface.png", "myface-detect.jpg", 1.01, 5)
    # HarrFaceDetect("myface1.png", "myface1-detect.jpg", 1.01, 1)
    # HarrFaceDetect("myface2.png", "myface2-detect.jpg", 1.01, 6)
    # HarrFaceDetect("testface1.png", "testface1-detect.jpg", 1.01, 5)
    HarrFaceDetect("myface3.png", "myface3-detect.jpg", 1.02, 5)