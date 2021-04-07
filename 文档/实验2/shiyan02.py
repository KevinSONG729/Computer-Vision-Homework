import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def readImg(path):
    img = cv.imread(path, 1)
    # cv.imshow('img', img) #显示图片1
    b, g, r = cv.split(img)
    img_plt = cv.merge([r, g, b])
    plt.imshow(img_plt, cmap='gray')
    # plt.show() #显示图片2
    cv.waitKey(0)
    return img


def detect(img):
    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    # cv.imshow("img_ycrcb", img_ycrcb)
    (y, cr, cb) = cv.split(img_ycrcb)
    cr1 = cv.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # cv.imshow("skin", skin)
    cv.waitKey(0)
    return skin


def display(origin_img, skin):
    b, g, r = cv.split(origin_img)
    origin_img = cv.merge([r, g, b])
    rows = origin_img.shape[0]
    cols = origin_img.shape[1]
    img = origin_img.copy()
    for i in range(0, rows):
        for j in range(0, cols):
            if skin[i, j] == 0:
                img[i, j] = (0, 0, 0)
    plt.subplot(131)
    plt.imshow(origin_img)
    plt.title("Original-Image")
    plt.xticks([]), plt.yticks([])
    plt.subplot(132)
    plt.imshow(skin, cmap="gray")
    plt.title("Mask")
    plt.xticks([]), plt.yticks([])
    plt.subplot(133)
    plt.imshow(img)
    plt.title("output")
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    print("shiyan02")
    img = readImg("friends.jpg")
    skin = detect(img)
    display(img, skin)
