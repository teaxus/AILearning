import numpy as np


def npDemo():
    print('''numpy的用法--------------------Begin''')
    data1 = np.array([1, 2, 3, 4, 5])
    print("data1：", data1, ",维度:", data1.shape)
    data2 = np.array([[1, 2], [3, 4]])
    print("data2：", data2, ",维度:", data2.shape)
    print("创建一个2*3，全部填充为0的数组", np.zeros([2, 3]))
    print("创建一个2*2，全部填充为1的数组", np.ones([2, 2]))
    # 改查
    data2[1, 0] = 5
    print("data2：", data2)
    print("data2[1,1]：", data2[1, 1])

    data3 = np.ones([2, 3])
    print("data3：", data3)
    print("data3*2：", data3 * 2)
    print("data3/2：", data3 / 2)
    print("data3+2：", data3 + 2)

    print("矩阵的加和乘")
    data4 = np.array([[1, 2, 3], [4, 5, 6]])
    print("data3+data4", data3 + data4)
    print("data3*data4", data3 * data4)
    print('''numpy的用法---End''')
