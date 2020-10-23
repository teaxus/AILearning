"""几何变化"""

import cv2
import numpy as np


# 通过opencv的API实现缩放功能
def resizeByApi():
    # 等比缩小一半
    dstHeight = int(src_height * 0.1)
    dstWidth = int(src_width * 0.8)
    dst = cv2.resize(src_img, (dstWidth, dstHeight))
    cv2.imshow('resizeByApiImage', dst)


# 使用矩阵运算方式缩放（使用最近临域插值法运算）
# src 10*20 dst 5*10
# dst <- src
# (1,2)  <-  (2,4)
# dst x 1  ->  src x 2 newX
# newX = x*(src 行/目标  行）  newX = 1*(10/5) = 2
# newY = y*(src 列/目标 列)  newY = 2*(20/10) = 4
# 12.3 = 2
def reizeByMatrixCal():
    img = cv2.imread('img/posExample/pos_0795.jpeg', 1)
    imgInfo = img.shape
    print(imgInfo)
    height = imgInfo[0]
    width = imgInfo[1]
    dstHeight = int(height / 2)
    dstWidth = int(width / 2)
    dstImage = np.zeros((dstHeight, dstWidth, 3), np.uint8)  # 0-255
    for i in range(0, dstHeight):  # 行
        for j in range(0, dstWidth):  # 列
            jNew = int(i * (height * 1.0 / dstHeight))
            iNew = int(j * (width * 1.0 / dstWidth))
            dstImage[i, j] = img[iNew, jNew]
    cv2.imshow('reizeByMatrixCalImg', dstImage)


# 图片的裁剪
def cutImg():
    dst = src_img[100:200, 100:300]
    cv2.imshow("cutImg", dst)


# 图片位移
def imgDisplacementByApi():
    matShift = np.float32([[1, 0, 100], [0, 1, 200]])  # 2*3建立移位矩阵
    # 对移位矩阵的理解
    # 第一步：先将[[1, 0, 100], [0, 1, 200]]拆分成两个矩阵，2*2的A矩阵（[[1,0],[0,1]]）和2*1的B矩阵（[[100],[200]]）
    # 第二步：假设C矩阵是图片矩阵（[[x],[y]]，1行2列图片矩阵）
    # 第三步：计算C*A+B = [[x*1+y*0],[0*x+1*y]]+[[100],[200]]
    # 第四步：使用推到结果，[[x+100],[y+200]]   例如：(10,20)   =>  (10+100,20+200)  =>  (110,220)
    dst = cv2.warpAffine(src_img, matShift, (src_height, src_width))
    cv2.imshow("imgDisplacementByApi", dst)


def imgDisplacementByMatrixCal():
    dst = np.zeros(src_img.shape, np.uint8)
    for i in range(0, src_height):
        for j in range(0, src_width - 100):
            dst[i, j + 100] = src_img[i, j]
    cv2.imshow("imgDisplacementByMatrixCal", dst)


src_img = cv2.imread('img/posExample/pos_0795.jpeg', 1)
src_img_info = src_img.shape
print(src_img_info)
src_height = src_img_info[0]
src_width = src_img_info[1]
src_mode = src_img_info[2]
cv2.imshow("imageSrc", src_img)

# 调用方法------Begin
# 缩放
resizeByApi()
reizeByMatrixCal()

# 裁剪
cutImg()

# 图片位移
imgDisplacementByApi()
imgDisplacementByMatrixCal()
# 调用方法------End

cv2.waitKey(0)
cv2.destroyAllWindows()
