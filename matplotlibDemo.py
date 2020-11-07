import matplotlib.pyplot as plt
import numpy as np


def matplotlibDemo():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([3, 5, 7, 6, 2, 6, 10, 15])
    plt.plot(x, y, 'r')
    plt.plot(x, y, 'g', lw=10)
    plt.show()
    # 折线 柱状
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([13, 25, 17, 36, 22, 26, 10, 15])
    plt.bar(x, y, 0.2, alpha=1, color='b')
    plt.show()

    # 类似股市的图片
    for i in range(0, 15):
        x_fi = np.zeros([2])
        x_fi[0] = i
        x_fi[1] = i

        y_fi = np.zeros([2])
        y_fi[0] = i + 10
        y_fi[1] = i + 20*i
        plt.plot(x_fi, y_fi, 'g', lw=8)

    plt.show()


if __name__ == '__main__':
    matplotlibDemo()
