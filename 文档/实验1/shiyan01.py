import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def readImg(path):
    img = cv.imread(path, 1)
    # cv.imshow('img', img) #显示图片1
    b,g,r = cv.split(img)
    img_plt = cv.merge([r,g,b])
    plt.imshow(img_plt, cmap='gray')
    # plt.show() #显示图片2
    # cv.waitKey(0)
    return img

def write_name(img):
    img_01 = cv.putText(img, '18121598', (10,50), cv.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (255, 255, 255), 4, cv.LINE_AA)
    cv.imshow('image_01', img_01)
    cv.imwrite('image_01.png', img_01)
    cv.waitKey(0)

def splite_img(img):
    img_1, img_2, img_3 = np.zeros(img.shape, dtype='uint8'), \
                          np.zeros(img.shape, dtype='uint8'), \
                          np.zeros(img.shape, dtype='uint8')
    channels = cv.split(img)
    img_1[:, :, 0],  img_2[:, :, 1],  img_3[:, :, 2] = channels[0], channels[1], channels[2]
    cv.imshow('b', img_1); cv.imwrite('b.png', img_1)
    cv.imshow('g', img_2); cv.imwrite('g.png', img_2)
    cv.imshow('r', img_3); cv.imwrite('r.png', img_3)
    cv.waitKey(0)

def edge(img):
    tree = img[191:505, 542:810]
    hsv = cv.cvtColor(tree, cv.COLOR_BGR2HSV)
    ret, binary = cv.threshold(hsv, 127, 255, cv.THRESH_BINARY)
    tree_edge = cv.inRange(hsv, (45, 85, 240), (124, 255, 255))
    rows, cols, channels = tree.shape
    for i in range(0, rows):
        for j in range(0, cols):
            if(tree_edge[i, j].all() == 0):
                img[195+i, 200+j] = tree[i, j]
    cv.imshow('result', img)
    cv.imwrite("result.png", img)
    cv.waitKey(0)

if __name__=="__main__":
    print("shiyan01")
    img = readImg('tree.jpg')
    # write_name(img)
    splite_img(img)
    # edge(img)
