import os
import initializers_plaidml
import keras as K
import cv2
import numpy as np

mnist = K.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = K.models.Sequential([
  K.layers.Flatten(input_shape=(28, 28)),
  K.layers.Dense(128, activation='relu'),
  K.layers.Dropout(0.2),
  K.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.save("./mnist_model.h5")
score = model.evaluate(x_test,  y_test, verbose=2)  # 返回的是损失值和你选定的指标值
print("score:{0}".format(score))

test_img = x_test[3]
cv2.imshow("first img show", test_img)
# 尝试使用test数据预测
predicted = model.predict(np.array([test_img], dtype=np.float32))
predicted_classes = model.predict_classes(np.array([test_img], dtype=np.float32))
print("\nPredicted softmax vector is: ")
print(predicted)    # 获取不同标签的概率
print(predicted_classes)  # 获取最高机会出现的概率
cv2.waitKey(0)

# from tensorflow.examples.tutorials.mnist import input_data
# import warnings
# warnings.filterwarnings('ignore')
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# batch_size = 100
# X_holder = tf.placeholder(tf.float32)
# y_holder = tf.placeholder(tf.float32)

# Weights = tf.Variable(tf.zeros([784, 10]))
# biases = tf.Variable(tf.zeros([1, 10]))
# predict_y = tf.nn.softmax(tf.matmul(X_holder, Weights) + biases)
# loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)

# session = tf.compat.v1.Session()
# init = tf.compat.v1.global_variables_initializer()
# session.run(init)

# for i in range(500):
#     images, labels = mnist.train.next_batch(batch_size)
#     session.run(train, feed_dict={X_holder: images, y_holder: labels})
#     if i % 25 == 0:
#         correct_prediction = tf.equal(
#             tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         accuracy_value = session.run(
#             accuracy, feed_dict={X_holder: mnist.test.images, y_holder: mnist.test.labels})
#         print('step:%d accuracy:%.4f' % (i, accuracy_value))
