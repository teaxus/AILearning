import tensorflow as tf


def fourOperations():
    data1 = tf.constant(6)  # 定义常量
    data2 = tf.Variable(2, name='var')  # 定义变量
    print(data1)
    print(data2)
    # 如果使用到变量，需要init
    init = tf.global_variables_initializer()

    print('''tf的四则运算---Begin''')
    dataAdd = tf.add(data1, data2)
    dataCopy = tf.assign(data2, dataAdd)  # dataAdd 赋值给 data2
    dataMul = tf.multiply(data1, data2)
    dataSub = tf.subtract(data1, data2)
    dataDiv = tf.divide(data1, data2)
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        print(sess.run(dataAdd))
        print(sess.run(dataMul))
        print(sess.run(dataSub))
        print(sess.run(dataDiv))

        print('sess.run(dataCopy)', sess.run(dataCopy))  # 8->data2
        print('dataCopy.eval()', dataCopy.eval())  # 8+6->14-data = 14   eval会自动找到一个sess，然后执行run操作
        print('tf.get_default_session()', tf.compat.v1.get_default_session().run(dataCopy))  # 和上面的操作是一样
    print('''tf的四则运算---End\n\n''')


def valPlaceholder():
    print("val placeholder begin")
    data1 = tf.compat.v1.placeholder(tf.float32)
    data2 = tf.compat.v1.placeholder(tf.float32)

    dataAdd = tf.add(data1, data2)
    with tf.compat.v1.Session() as sess:
        print(sess.run(dataAdd, feed_dict={data1: 6, data2: 2}))
        # 1 data Add 2 data(feed_dict = data1：6，data2：2})
    print("val placeholder end\n\n")
