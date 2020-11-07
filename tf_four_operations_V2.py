import tensorflow as tf

def fourOperations():
    data1 = tf.constant(6)  # 定义常量
    data2 = tf.Variable(2, name='var')  # 定义变量
    print(data1)
    print(data2)
    # 如果使用到变量，需要init
    # init = tf.global_variables_initializer()

    print('''tf的四则运算---Begin''')
    dataAdd = tf.add(data1, data2)
    dataCopy = tf.assign(data2, dataAdd)  # dataAdd 赋值给 data2
    dataMul = tf.multiply(data1, data2)
    dataSub = tf.subtract(data1, data2)
    dataDiv = tf.divide(data1, data2)
    print('''tf的四则运算---End\n\n''')


def valPlaceholder():
    print("val placeholder begin")
    data1 = tf.placeholder(tf.float32)
    data2 = tf.placeholder(tf.float32)

    dataAdd = tf.add(data1, data2)
    with tf.Session() as sess:
        print(sess.run(dataAdd, feed_dict={data1: 6, data2: 2}))
        # 1 data Add 2 data(feed_dict = data1：6，data2：2})
    print("val placeholder end\n\n")


if __name__ == '__main__':
    # fourOperations()
    valPlaceholder()
