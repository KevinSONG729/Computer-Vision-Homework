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
    flag = 0

def getKeyValue(item):
    key = item.value
    return key

def getKeyValue2(item):
    key = item[0]
    return key

def getData():
    images = []
    image_test = []
    filename = []
    for i in range(40):
        filename.append("s" + str(i+1))
    for img_dir in filename:
        for img in os.listdir("D:/专业课/计算机视觉/文档/实验3/att_faces/" + img_dir):
            image = cv.imread("att_faces/" + img_dir + "/" + img, 0)
            image = (np.array(image).reshape(1, -1)).tolist()
            if (img != "10.pgm"):
                # print(img)
                images.append(image[0])
            else:
                image_test.append(image[0])
    images = np.array(images)
    image_test = np.array(image_test)
    image_submean = (images.copy()).astype(float)
    image_mean = np.mean(images, axis = 0)
    # print(type(image_mean[0]))
    print(image_mean)
    for i in range(len(image_submean)):
        image_submean[i] = [x-y for x, y in zip(image_submean[i], image_mean)]
    # print("image_sub")
    # print(image_submean)
    image_mean = np.array(image_mean).reshape(112, 92)
    plt.imshow(image_mean, cmap="gray")  # 平均脸
    # plt.show()
    image_mean = image_mean.reshape(1, -1)
    return images, image_submean, image_test, image_mean

def PCA(images, image_submean):
    fea_val_vec = []
    fea_val_vec_all = []
    res = []
    res_all = []
    cov1 = np.cov(images, rowvar = False) #(10304, 10304)
    cov2 = np.dot(image_submean, np.transpose(image_submean))
    # print(type(cov2))
    # cov2 = np.cov(images, rowvar = True)  #(360, 360)
    # a1, b1 = linalg.eig(cov1)
    a2, b2 = linalg.eig(cov2)
    b2 = np.transpose(b2)
    # b1 = np.transpose(b1).astype(float)
    # for i in range(len(a1)):
    #     fea = sxw()
    #     fea.value = a1[i]
    #     fea.vector = b1[i]
    #     fea.flag = int(i / 9 + 1)
    #     fea_val_vec_all.append(fea)
    # fea_val_vec_all = sorted(fea_val_vec_all, key = getKeyValue, reverse = True)
    # for i in fea_val_vec_all:
    #     res_all.append(i.vector)
    # res_all = np.array(res_all)
    for i in range(len(a2)):
        fea = sxw()
        fea.value = a2[i]
        fea.vector = b2[i]
        fea.flag = int(i/9 + 1)
        fea_val_vec.append(fea)
    fea_val_vec_copy = fea_val_vec.copy()
    fea_val_vec = sorted(fea_val_vec, key = getKeyValue, reverse = True)
    # fea_val_vec = fea_val_vec[0:350]
    for i in fea_val_vec:
        res.append(i.vector) # (42, 360)
    res = np.array(res)
    res = res[0:42]
    # print(res.shape)
    res_feature_vec = np.dot(res, image_submean) # (42, 10304)
    res = np.transpose(res) # (360, 42)
    print(image_submean.shape)
    res = np.dot(images, np.transpose(res_feature_vec))
    # print(res)
    # print(res_feature_vec)
    return res, res_feature_vec, fea_val_vec_copy, fea_val_vec, res_all

def opencv_eigenface(x, test):
    accuracy = 0
    print(test.shape)
    y = [int(i/9 + 1) for i in range(len(x))]
    y = np.array(y)
    model = cv.face.EigenFaceRecognizer_create(42)
    model.train(x, y)
    for i in range(len(test)):
        label, confidence = model.predict(test[i])
        if(label == i + 1):
            accuracy = accuracy + 1
    accuracy = accuracy / len(test)
    print("opencv 预测正确率为：" + str(accuracy))

def plot(res):
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(len(res)):
        plt.subplot(6, 7, i+1)
        image = res[i].reshape(112, 92)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title("Face " + str(i+1))
    plt.show()

def Knn(res, test, res_feature_vec, fea_val_vec_copy, fea_val_vec):
    accuracy = 0
    # print(res.shape)
    test = np.dot(test, np.transpose(res_feature_vec))
    # print(test.shape)
    sample = []
    sample_test = []
    # print(res.shape)
    # print("res 360 * 42")
    for i in range(len(res)):
        # print(res[i])
        x = sxw()
        x.value = int(i/9 + 1)
        x.vector = res[i]
        sample.append(x)
    # print(test.shape)
    # print("test")
    for i in range(len(test)):
        dis = []
        for j in range(len(sample)):
            p = 0
            for k in range(len(sample[j].vector)):
                p = p + (sample[j].vector[k]-test[i][k])*(sample[j].vector[k]-test[i][k])
            # print(p)
            dis.append([np.sqrt(p), int(j/9+1)])
        dis = sorted(dis, key = getKeyValue2, reverse = False)
        # print(dis)
        if(dis[0][1] == i + 1):
            print("第" + str(i + 1) + "类预测正确！")
            accuracy = accuracy + 1
        else:
            print("第" + str(i + 1) + "类预测错误！")
    accuracy = accuracy / len(test)
    print("accuracy = " + str(accuracy))

def construct(res_feature_vec, image_mean, image, b1):
    b2 = b1[0:8633]
    b3 = b1[0:7006]
    b4 = b1[0:5376]
    b5 = b1[0:3752]
    b6 = b1[0:2125]
    b7 = b1[0:490]
    b8 = b1[0:360]
    b9 = b1[0:200]
    b10 = b1[0:100]
    b11 = b1[0:50]
    b = [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11]
    for i in b:
        r = np.transpose(np.dot(image, np.transpose(i)))
        constructed_image = image_mean + np.dot(np.transpose(r), i)
        constructed_image = constructed_image.reshape(112, 92)
        plt.imshow(constructed_image, cmap="gray")
        plt.show()


def plot2():
    num = [50, 100, 200, 360, 490, 2125, 3752, 5376, 7006, 8633, 10304]
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(0, 11):
        plt.subplot(3, 4, i + 1)
        image = cv.imread(str(i+1) + ".png", 0)
        image = np.array(image)
        print(image)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title("Face " + str(i+1) + "(" + str(num[i]) + ")")
    plt.subplot(3, 4, 12)
    image = cv.imread("10.pgm", 0)
    image = np.array(image)
    print(image)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.title("Destination Face ")
    plt.show()

if __name__ == "__main__":
    print("shiyan03")
    # images, image_submean, image_test, image_mean = getData()
    # res, res_feature_vec, fea_val_vec_copy, fea_val_vec, b1 = PCA(images, image_submean)
    # opencv_eigenface(images, image_test)
    # plot(res_feature_vec)
    # Knn(res, image_test, res_feature_vec, fea_val_vec_copy, fea_val_vec)
    # construct(res_feature_vec, image_mean, image_test[2], b1)
    plot2()