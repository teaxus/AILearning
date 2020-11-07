from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.datasets import mnist
import cv2
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

(X_train, label_train), (X_test, label_test) = mnist.load_data()
# image = X_train[0]
# fig = plt.figure
# plt.imshow(image, cmap='gray')
# plt.show()


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
images, labels = mnist.train.next_batch(100)

cv2.imshow("reizeByMatrixCalImg", numpy.array(list_split(images[0], 28)))
# for i in range(10):
#     cv2.imshow("reizeByMatrixCalImg_{0}".format(i), X_train[i])


cv2.waitKey(0)
cv2.destroyAllWindows()
