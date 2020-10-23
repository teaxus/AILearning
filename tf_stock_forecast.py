# 股票预测
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

date = np.linspace(1, 15, 15)
endPrice = np.array(
    [2511.90, 2538.26, 2510.68, 2591.66, 2732.98, 2701.69, 2701.29, 2678.67, 2726.50, 2681.50, 2739.17, 2715.07,
     2823.58, 2864.90, 2919.08])
beginPrice = np.array(
    [2438.71, 2500.88, 2534.95, 2512.52, 2594.02, 2743.26, 2697.47, 2697.47, 2695.24, 2678.23, 2722.13, 2674.93,
     2717.47, 2832.73, 2877.40])
print(date)
plt.figure()

for i in range(0, 15):
    # 柱状图
    dateOne = np.zeros([2])
    dateOne[0] = i
    dateOne[1] = i

    priceOne = np.zeros([2])
    priceOne[0] = beginPrice[i]
    priceOne[1] = endPrice[i]
    if endPrice[i] > beginPrice[i]:
        plt.plot(dateOne, priceOne, 'r', lw=8)
    else:
        plt.plot(dateOne, priceOne, 'g', lw=8)
# plt.show()


# A(15x1)*w1(1x10)+b1(1*10) = B(15x10)
# B(15x10)*w2(10x1)+b2(15x1) = C(15x1)
# 1 A(输入层15行x1列) B（隐藏层1行x10列） C（输出层15行x1列）
dateNormal = np.zeros([15, 1])
priceNormal = np.zeros([15, 1])

# 归一化处理
for i in range(0, 15):
    dateNormal[i] = i / 14.0
    priceNormal[i] = endPrice[i] / 3000.0
x = tf.placeholder(tf.float32, [None, 1])  # A阵列（输入层）
y = tf.placeholder(tf.float32, [None, 1])  # C阵列（输出层）

# 开始神经网络的搭建
# B
w1 = tf.Variable(tf.random_uniform([1, 10], 0, 1))  # 创建一个1行10列的权重阵列，随机填充0到1的数据
b1 = tf.Variable(tf.zeros([1, 10]))
wb1 = tf.matmul(x, w1) + b1
layer1 = tf.nn.relu(wb1)  # 激励函数

# C
w2 = tf.Variable(tf.random_uniform([10, 1], 0, 1))  # 创建一个10行1列的权重阵列，随机填充0到1的数据
b2 = tf.Variable(tf.zeros([15, 1]))
wb2 = tf.matmul(layer1, w2) + b2
layer2 = tf.nn.relu(wb2)  # 激励函数
loss = tf.reduce_mean(tf.square(y - layer2))  # 计算差异值（y 真实layer2计算，相当于方差）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 使用梯度梯度下降法减少loss

# 完成神经网络的搭建


# 开始训练
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())  # 初始化变量
    for i in range(0, 1000):
        # 训练
        sess.run(train_step, feed_dict={x: dateNormal, y: priceNormal})
    # w1w2 b1b2 A + wb --> layer2
    # 展示预估结果
    pred = sess.run(layer2, feed_dict={x: dateNormal})
    predPrice = np.zeros([15, 1])
    for i in range(0, 15):
        predPrice[i, 0] = (pred * 3000)[i, 0]
    plt.plot(date, predPrice, 'b', lw=1)
plt.show()
