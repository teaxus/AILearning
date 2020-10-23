# coding:utf8

import cv2


def pxOperation():
    img = cv2.imread('img/posExample/pos_0795.jpeg', 1)
    (b, g, r) = img[100, 100]  # 读取单个bgr（rgb）单元数值
    print('rgb is', r, g, b)
    width = len(img)
    height = len(img[0])
    print('width is:', width, '\nheigh is:', height)

    # 遍历行像素
    for x in range(0, width):
        # 遍历列元素
        for y in range(0, height):
            if y > 1.2 * x:
                (b, g, r) = img[x, y]
                img[x, y] = (r * r % 255, g * g % 255, b * b % 255)

    cv2.imwrite('output/img/像素操作.jpeg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])  # 设置图片质量

    cv2.imshow('imgage', img)
    cv2.waitKey(0)
