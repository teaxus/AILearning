# coding=utf-8
# hdf5_load_save_model_keras.py
# ========================================================
# 模型的加载
import initializers_plaidml
from iris_keras_dnn import main
from iris_keras_dnn import load_iris_data
import h5py
import numpy as np
from keras.models import load_model

model_path = "./iris_model.h5"
# main(model_path)
# 训练结果
# Evaluation on test data: loss = 0.075852 accuracy = 97.78%
# Using model to predict species for features:
# [[6.1 3.1 5.1 1.1]]
# Predicted softmax vector is:
# [[3.2467e-05 9.9274e-01 7.2239e-03]]
# Predicted species is:
# Iris-versicolor

train_x, test_x, train_y, test_y, Class_dict = load_iris_data()


def loadDataWithH5PY():
    with h5py.File(model_path, 'r') as f:
        dense_1 = f['/model_weights/dense_1/dense_1']
        dense_1_bias = dense_1['bias'][:]
        dense_1_kernel = dense_1['kernel'][:]

        dense_2 = f['/model_weights/dense_2/dense_2']
        dense_2_bias = dense_2['bias'][:]
        dense_2_kernel = dense_2['kernel'][:]

        dense_3 = f['/model_weights/dense_3/dense_3']
        dense_3_bias = dense_3['bias'][:]
        dense_3_kernel = dense_3['kernel'][:]

        print("第一层的连接权重矩阵：\n%s\n" % dense_1_kernel)
        print("第一层的连接偏重矩阵：\n%s\n" % dense_1_bias)
        print("第二层的连接权重矩阵：\n%s\n" % dense_2_kernel)
        print("第二层的连接偏重矩阵：\n%s\n" % dense_2_bias)
        print("第三层的连接权重矩阵：\n%s\n" % dense_3_kernel)
        print("第三层的连接偏重矩阵：\n%s\n" % dense_3_bias)


        # 模拟每个神经层的计算，得到该层的输出
        def layer_output(input, kernel, bias):
            return np.dot(input, kernel) + bias


        # 实现ReLU函数
        relu = np.vectorize(lambda x: x if x >= 0 else 0)


        # 实现softmax函数
        def softmax_func(arr):
            exp_arr = np.exp(arr)
            arr_sum = np.sum(exp_arr)
            softmax_arr = exp_arr/arr_sum
            return softmax_arr


        # 输入向量
        unkown = np.array([[6.1, 3.1, 5.1, 1.1]], dtype=np.float32)

        # 第一层的输出
        print("模型计算中...")
        output_1 = layer_output(unkown, dense_1_kernel, dense_1_bias)
        output_1 = relu(output_1)

        # 第二层的输出
        output_2 = layer_output(output_1, dense_2_kernel, dense_2_bias)
        output_2 = relu(output_2)

        # 第三层的输出
        output_3 = layer_output(output_2, dense_3_kernel, dense_3_bias)
        output_3 = softmax_func(output_3)

        # 最终的输出的softmax值
        np.set_printoptions(precision=4)
        print("最终的预测值向量为: %s" % output_3)


def loadDataWithKerasModel():
    print("Using loaded model to predict...")
    model = load_model(model_path)
    np.set_printoptions(precision=4)
    unknown = np.array([[6.1, 3.1, 5.1, 1.1]], dtype=np.float32)
    predicted = model.predict(unknown)
    print("Using model to predict species for features: ")
    print(unknown)
    print("\nPredicted softmax vector is: ")
    print(predicted)
    species_dict = {v: k for k, v in Class_dict.items()}
    print("\nPredicted species is: ")
    print(species_dict[np.argmax(predicted)])


if __name__ == '__main__':
    loadDataWithH5PY()
    loadDataWithKerasModel()
