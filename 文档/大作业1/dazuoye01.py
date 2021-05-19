import cv2 as cv
import os
from scipy import linalg
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist

def dataEnhancement():
    img = cv.imread('1-a.png', 0)
    w, h = img.shape[:2]
    center = (w // 2, h // 2)
    for i in np.arange(0.0, 360.0, 1):
        M = cv.getRotationMatrix2D(center, i, 1)
        img_rotated = cv.warpAffine(img, M, (w, h), borderValue=239)
        cv.imwrite('img_rotated/'+str(i)+'_rotate.png', img_rotated)

def detect():
    best_point_list = []
    flag = 1
    img = cv.imread('1-b.png', 0)
    for path in np.arange(0.0, 360.0, 1):
        min_dis = float('inf')
        best_point = (0, 0)
        img_template = cv.imread('img_rotated/' + str(path) + '_rotate.png', 0)
        img_template = cv.threshold(img_template, 128, 255, cv.THRESH_BINARY)[1]
        img_template = img_template.reshape(-1,)
        for h in range(0,110,4):
            for w in range(0,260,4):
                img_detect = img[h : h+40, w : w+40]
                img_detect = cv.threshold(img_detect, 128, 255, cv.THRESH_BINARY)[1]
                img_detect = img_detect.reshape(-1,)
                # dis = np.sqrt(sum(map(sum, (img_template - img_detect) ** 2)))
                X = np.vstack([img_detect, img_template])
                dis = pdist(X)[0]
                # dis = 0
                # for i in (img_detect-img_template)**2:
                #     dis = dis + i
                # dis = dis**0.5
                # print(path + str(dis))
                if(dis < min_dis):
                    min_dis = dis
                    best_point = (h, w)
        if(best_point != (0, 0)):
            h_best = best_point[0]
            w_best = best_point[1]
            if best_point_list!=[]:
                for i in best_point_list:
                    flag = 1
                    if (best_point[0]>i[0]-9 and best_point[0]<i[0]+9 and best_point[1]>i[1]-9 and best_point[1]<i[1]+9):
                        flag = 0
                        break
                if(flag):
                    best_point_list.append(best_point)
                    # img_best = img[h_best:h_best + 40, w_best:w_best + 40]
                    # plt.imshow(img_best)
                    # plt.show()
                    print(str(path) + '_rotate' + ' find best in' + str(best_point) + " dis= " + str(min_dis))
            else:
                best_point_list.append(best_point)
                # img_best = img[h_best:h_best + 40, w_best:w_best + 40]
                # plt.imshow(img_best)
                # plt.show()
                print(str(path) + '_rotate' + ' find best in' + str(best_point) + " dis= " + str(min_dis))
    print(best_point_list)
    for i in best_point_list:
        cv.rectangle(img, (i[1], i[0]), (i[1]+40, i[0]+40), (0, 255, 0), 1)
    cv.imwrite('detect_img/result1.png', img)
    cv.waitKey()

def SIFT(img):
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    # img_a = cv.drawKeypoints(img_a, kp, img_a, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(img_a)
    # plt.show()

    # if(len(kp)!=0):
    #     des = des.reshape(-1,)

    return des, kp

def SIFT_detect():
    img_a = cv.imread('1-a.png', 0)
    feature_a, kp_a = SIFT(img_a)
    print(len(kp_a))
    print(feature_a)
    img_b = cv.imread('1-b.png', 0)
    img_b_copy = img_b.copy()
    img_list = []
    for h in range(0, 110, 3):
        for w in range(0, 260, 3):
            img_detect = img_b[h: h + 40, w: w + 40]
            feature_b, kp_b = SIFT(img_detect)
            flag = 1
            if len(kp_b) == 18:
                # print('kp_b == 18')
                bf = cv.BFMatcher(cv.NORM_L2)
                matched = bf.knnMatch(feature_a, feature_b, k=2)
                goodMatch = []
                for m, n in matched:
                    if m.distance < n.distance:
                        goodMatch.append(m)
                if (len(goodMatch) == 18):
                    print("goodmatch" + str(len(goodMatch)))
                    if img_list != []:
                        for i in range(len(img_list)):
                            if (h > img_list[i][0] - 18 and h < img_list[i][0] + 18  and w > img_list[i][1] - 24 and
                                    w < img_list[i][1] + 24):
                                print('cfl')
                                flag = 0
                                print((h+img_list[i][0])/2,' ',(w+img_list[i][1])/2)
                                img_list[i] = ((h+img_list[i][0])/2, (w+img_list[i][1])/2)
                                break
                        if flag:
                            # img_detect_final = cv.drawMatches(img_a, kp_a, img_detect, kp_b, goodMatch, None, flags=2)
                            # plt.imshow(img_detect_final)
                            # plt.show()
                            img_list.append((h, w))
                    else:
                        # img_detect_final = cv.drawMatches(img_a, kp_a, img_detect, kp_b, goodMatch, None, flags=2)
                        # plt.imshow(img_detect_final)
                        # plt.show()
                        img_list.append((h, w))
    print(img_list)
    for i in img_list:
        print(i[1], i[0])
        cv.rectangle(img_b_copy, (int(i[1]), int(i[0])), (int(i[1]+35), int(i[0]+35)), (0, 100, 0), 1)
    cv.imshow("img", img_b_copy)
    cv.waitKey()


def get_myway_dis(img):
    edges = cv.Canny(img, 0, 128)
    edges = edges.reshape(-1, )
    edges_list = []
    dis = 0
    for i in range(len(edges)):
        if edges[i] == 255:
            dis += (i//20 - 19) ** 2 + (i%20 -19) ** 2
    dis = dis ** 0.5
    return dis


def myway():
    img_a = cv.imread('1-a.png', 0)
    img_b = cv.imread('1-b.png', 0)
    dis = get_myway_dis(img_a[9:31, 9:31])
    img_b_copy = img_b.copy()
    best_fit = []
    for h in range(0, 110, 3):
        for w in range(0, 260, 3):
            flag = 1
            img_detect = img_b[h : h + 22, w : w + 22]
            dis_detect = get_myway_dis(img_detect)
            if(np.abs(dis_detect - dis) <= 3.5):
                if best_fit != []:
                    for i in range(len(best_fit)):
                        if (h > best_fit[i][0] - 18 and h < best_fit[i][0] + 18 and w > best_fit[i][1] - 24 and
                                w < best_fit[i][1] + 24):
                            print('cfl')
                            flag = 0
                            print((h + best_fit[i][0]) / 2, ' ', (w + best_fit[i][1]) / 2)
                            best_fit[i] = ((h + best_fit[i][0]) / 2, (w + best_fit[i][1]) / 2)
                            break
                    if flag:
                        best_fit.append((h, w))
                else:
                    best_fit.append((h, w))
    print(best_fit)
    for i in best_fit:
        cv.rectangle(img_b_copy, (int(i[1]-9), int(i[0]-9)), (int(i[1]+31), int(i[0]+31)), (0, 255, 0), 1)
    cv.imwrite('detect_img/result2.png', img_b_copy)
    cv.waitKey()


if __name__ == "__main__":
    print("dazuoye01")
    # dataEnhancement()
    # detect()
    # SIFT_detect()
    myway()
