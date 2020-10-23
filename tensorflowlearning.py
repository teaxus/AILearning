# coding:utf8

# tf基础
# 基础数据类型  运算符  流程  字典 数组
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tf_four_operations import fourOperations
from tf_four_operations import valPlaceholder
from tf_matrix import makeMatrix
from tf_matrix import matrixOperation
from np_matrix import npDemo
from matplotlibDemo import matplotlibDemo

'''基础功能--------------------Begin'''
fourOperations()  # 四则运算
valPlaceholder()  # 变量的placeholder
'''基础功能---End'''

'''矩阵功能--------------------Begin'''
makeMatrix()  # 创建矩阵
matrixOperation()  # 矩阵的加乘
npDemo()  # numpy的demo功能（和matrixOperation里面的差不多）
'''矩阵功能---End'''


'''matplotlib的demo--------------------Begin'''
matplotlibDemo()
'''matplotlib的demo---End'''

