import cv2 as cv
import os
from scipy import linalg
import numpy as np
from matplotlib import pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class sxw:
    value = 0
    vector = []

def getKeyValue(item):
    key = item.value
    return key

def getData():
    img_list = []
    label = []
    for i in range(10):
        for img in os.listdir("D:/专业课/计算机视觉/文档/实验4/number_data/training/" + str(i)):
            image = cv.imread("number_data/training/" + str(i) + "/" + img, 0)
            width = image.shape[0]
            height = image.shape[1]
            image = image[int(width/2) - 25: int(width/2) + 25, int(height/2) - 25: int(height/2) + 25]
            image = (np.array(image).reshape(1,-1)).astype(float)
            img_list.append(image[0])
            label.append(i)
    img_list, label = np.array(img_list), np.array(label)
    return img_list, label

def getTest():
    test_list = []
    test_label = []
    for i in range(10):
        for img in os.listdir("D:/专业课/计算机视觉/文档/实验4/number_data/testing/" + str(i)):
            image = cv.imread("number_data/testing/" + str(i) + "/" + img, 0)
            width = image.shape[0]
            height = image.shape[1]
            image = image[int(width / 2) - 25: int(width / 2) + 25, int(height / 2) - 25: int(height / 2) + 25]
            image = (np.array(image).reshape(1, -1)).astype(float)
            test_list.append(image[0])
            test_label.append(i)
    test_list, test_label = np.array(test_list), np.array(test_label)
    return test_list, test_label

def PCA(imgs, test):
    imgs_copy = imgs.copy()
    fea_val_vec = []
    mean = np.mean(imgs, axis=0)
    for i in range(len(imgs)):
        imgs[i] = imgs[i] - mean
    imgs = np.transpose(imgs)
    cov = np.dot(imgs, np.transpose(imgs))
    a1, b1 = linalg.eig(cov)
    a1 = a1.astype(float)
    sum = np.sum(a1)
    b1 = np.transpose(b1).astype(float)
    for i in range(len(a1)):
        fea = sxw()
        fea.value = a1[i]
        fea.vector = b1[i]
        fea_val_vec.append(fea)
    fea_val_vec = sorted(fea_val_vec, key = getKeyValue, reverse = True)
    fea_val_vec = fea_val_vec[0:20]
    res = []
    for i in fea_val_vec:
        res.append(i.vector)
    res = np.array(res)
    imgs_jw = np.dot(res, np.transpose(imgs_copy)).astype(np.float32)
    imgs_test_jw = np.dot(res, np.transpose(test)).astype(np.float32)
    return imgs_jw, imgs_test_jw # 一列是一个样本

def Knn(imgs, labels, imgs_test, labels_test):
    print(imgs_test.shape)
    knn = cv.ml.KNearest_create()
    knn.train(np.transpose(imgs), cv.ml.ROW_SAMPLE, labels)
    ret, result, neighbours, dist = knn.findNearest(np.transpose(imgs_test), k=5)
    result = result.reshape(1,-1).tolist()[0]
    accuracy = 0
    for i in range(len(result)):
        if result[i] == labels_test[i]:
            accuracy = accuracy + 1
    accuracy = accuracy/len(result)
    print(labels_test)
    print(result)
    print(accuracy)

def hog(imgs, imgs_test):
    imgs = imgs.astype(np.uint8)
    imgs_test = imgs_test.astype(np.uint8)
    print(imgs.shape)
    winSize = (50, 50)
    blockSize = (32, 32)
    blockStride = (9, 9)
    cellSize = (8, 8)
    nbins = 9
    padding = (8, 8)
    winStride = (0, 0)
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    print(imgs[0].shape)
    hog_matrix = []
    hog_matrix_test = []
    for i in imgs:
        test_hog = hog.compute(i.reshape(50, 50), winStride, padding)
        test_hog = test_hog.reshape(1, -1).tolist()
        hog_matrix.append(test_hog[0])
    hog_matrix = np.array(hog_matrix)
    for i in imgs_test:
        test_hog = hog.compute(i.reshape(50, 50), winStride, padding)
        test_hog = test_hog.reshape(1, -1).tolist()
        hog_matrix_test.append(test_hog[0])
    hog_matrix_test = np.array(hog_matrix_test)
    return hog_matrix, hog_matrix_test

def svm(imgs, labels, imgs_test, labels_test):
    imgs = imgs.astype(np.float32)
    imgs_test = imgs_test.astype(np.float32)
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.train(imgs, cv.ml.ROW_SAMPLE, labels)
    print(imgs_test[0].shape)
    a, b  = svm.predict(np.matrix(imgs_test))
    b = np.transpose(b).tolist()[0]
    accuracy = 0
    for i in range(len(labels_test)):
        if(labels_test[i] == b[i]):
            accuracy = accuracy + 1
    accuracy = accuracy/len(labels_test)
    print(accuracy)


if __name__ == "__main__":
    print("shiyan04")
    imgs, labels = getData()
    imgs_test, labels_test = getTest()
    # imgs_jw, imgs_test_jw = PCA(imgs, imgs_test)
    # Knn(imgs_jw, labels, imgs_test_jw, labels_test)
    hog_imgs, hog_imgs_test = hog(imgs, imgs_test)
    svm(hog_imgs, labels, hog_imgs_test, labels_test)
