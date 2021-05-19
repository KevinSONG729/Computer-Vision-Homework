import cv2 as cv
import os
from scipy import linalg
import numpy as np
from matplotlib import pyplot as plt

def plot_detect():
    img_list = []
    for path in os.listdir('detect_img'):
        img_list.append(cv.imread('detect_img/' + path, 0))
    for i in range(len(img_list)):
        plt.subplot(3, 4, i+1)
        plt.imshow(img_list[i])
        plt.axis('off')
    plt.imsave('detect_img/result.png')

if __name__ == '__main__':
    print('plot')
    plot_detect()