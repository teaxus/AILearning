import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()

# 创建矩阵
def makeMatrix():
    # 矩阵  数组 M行N列  【】   内部也是【】表示     【里面 列数据】    【】中括号 行数
    # [[6,6]]  （一行两列数据）
    print("矩阵的定义---Begin")
    data1 = tf.constant([[6, 6]])
    data2 = tf.constant([
        [2],
        [2]
    ])
    data3 = tf.constant([[3, 3]])
    data4 = tf.constant([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    mat1 = tf.constant([[2], [3], [4]])
    print(data4.shape)  # 打印矩阵维度（结果是3行2列）
    with tf.compat.v1.Session() as sess:
        print("打印data4的整体", data4.eval())
        print("打印第一行", sess.run(data4[0]))
        print("打印第0列", sess.run(data4[:, 0]))
        print("打印第0行，第1列数据", sess.run(data4[0, 1]))

        print("创建4*5的全零矩阵", tf.zeros([4, 5]).eval())
        print("创建5*4的全一矩阵", tf.ones([5, 4]).eval())
        print("创建5*4的矩阵，填充数值统一是18", tf.fill([5, 4], 18).eval())
        print("mat1矩阵：", mat1.eval())
        print("创建一个和mat1格式一样的矩阵，并且内容全部填充为0", tf.zeros_like(mat1).eval())
        print("创建一个数值从0到2，平均分成10份的数组（因为从0开始，所以算11）", tf.linspace(0.0, 2.0, 11).eval())
        print("生成一个5*6的数组，内容随机", tf.random.uniform([5 * 6], -1, 2).eval())
    print("矩阵的定义---End\n\n")


# 矩阵的加乘
def matrixOperation():
    print('''矩阵加乘--------------------Begin''')
    data1 = tf.constant([[5, 6]])
    data2 = tf.constant([
        [2],
        [3]
    ])
    data3 = tf.constant([[3, 3]])

    matMul = tf.matmul(data1, data2)
    matMul2 = tf.multiply(data1, data3)
    matAdd = tf.add(data1, data3)

    with tf.compat.v1.Session() as sess:
        print("data1：", data1.eval())
        print("data2：", data2.eval())
        print("data3：", data3.eval())
        print("开始计算")
        print("这种乘法是（data1的5 * data2的2） + （data1的6 * data2的3）= 28；就是第一个数据每行，乘每列后相加，matMul：",
              matMul.eval())
        print("这种乘法是[（data1的5 * data3的3）= 15 ,（data1的6 * data3的3）= 18 ]  ==>  [15,18]，matMul2：",
              matMul2.eval())
        print("这种加法是[（data1的5 + data3的3）= 8 ,（data1的6 * data3的3）= 9 ]  ==>  [8,9]，matAdd：",
              matAdd.eval())
    print('''矩阵加乘---End\n\n''')


# numpy的demo功能（和matrixOperation里面的差不多）
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

if __name__ == '__main__':
    makeMatrix()
    matrixOperation()
    npDemo()