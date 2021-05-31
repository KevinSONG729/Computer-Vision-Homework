# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
from cv2 import Stitcher

if __name__ == "__main__":
    img1_1 = cv2.imread('two/Hanging/Hanging1.png')
    img1_2 = cv2.imread('two/Hanging/Hanging2.png')
    img1_2 = cv2.rotate(img1_2, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img2_1 = cv2.imread("two/uttower/uttower_left.jpg")
    img2_2 = cv2.imread("two/uttower/uttower_right.jpg")

    img3_1 = cv2.imread("multi/pier/1.jpg")
    img3_2 = cv2.imread("multi/pier/2.jpg")
    img3_3 = cv2.imread("multi/pier/3.jpg")

    img4_1 = cv2.imread("multi/Rainier/Rainier1.png")
    img4_2 = cv2.imread("multi/Rainier/Rainier2.png")
    img4_3 = cv2.imread("multi/Rainier/Rainier3.png")
    img4_4 = cv2.imread("multi/Rainier/Rainier4.png")
    img4_5 = cv2.imread("multi/Rainier/Rainier5.png")
    img4_6 = cv2.imread("multi/Rainier/Rainier6.png")

    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    (_result1, pano1) = stitcher.stitch((img1_1, img1_2))
    (_result2, pano2) = stitcher.stitch((img2_1, img2_2))
    (_result3, pano3) = stitcher.stitch((img3_1, img3_2, img3_3))
    (_result4, pano4) = stitcher.stitch((img4_1, img4_2, img4_3, img4_4, img4_5, img4_6))
    res1 = cv2.resize(pano1, (1000, 600), interpolation=cv2.INTER_CUBIC)
    res2 = cv2.resize(pano2, (1000, 600), interpolation=cv2.INTER_CUBIC)
    res3 = cv2.resize(pano3, (1000, 600), interpolation=cv2.INTER_CUBIC)
    res4 = cv2.resize(pano4, (1000, 600), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite("result1.png", res1)
    cv2.imwrite("result2.png", res2)
    cv2.imwrite("result3.png", res3)
    cv2.imwrite("result4.png", res4)


