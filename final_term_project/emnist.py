from os.path import dirname, join as pjoin
import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np

from sklearn import mixture

import tensorflow as tf
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
emnist = spio.loadmat("emnist-balanced.mat")

x_train = np.asarray(emnist["dataset"][0][0][0][0][0][0])
np.random.shuffle(x_train)
y_train = emnist["dataset"][0][0][0][0][0][1]

x_train = x_train / 255.0
x_trainer = x_train.reshape(x_train.shape[0], 28 * 28)

dpgmm = mixture.GaussianMixture(n_components=100,
                                        covariance_type='full', random_state=12).fit(x_trainer[:5000])
predicted = dpgmm.predict(x_trainer[:5000])

components = list([] for i in range(100))


num_0 = 1
num_1 = 1
num_2 = 1
num_3 = 1
num_4 = 1
num_5 = 1
num_6 = 1
for i in range(5000):
    components[predicted[i]].append(y_train[i])
    if predicted[i] == 0:
        plt.imshow(x_train[i].reshape(28, 28))
        plt.savefig("0-" + str(num_0) + ".png")
        plt.clf()
        num_0 += 1
    if predicted[i] == 1:
        plt.imshow(x_train[i].reshape(28, 28))
        plt.savefig("1-" + str(num_1) + ".png")
        plt.clf()
        num_1 += 1
    if predicted[i] == 2:
        plt.imshow(x_train[i].reshape(28, 28))
        plt.savefig("2-" + str(num_2) + ".png")
        plt.clf()
        num_2 += 1
    if predicted[i] == 3:
        plt.imshow(x_train[i].reshape(28, 28))
        plt.savefig("3-" + str(num_3) + ".png")
        plt.clf()
        num_3 += 1
    if predicted[i] == 4:
        plt.imshow(x_train[i].reshape(28, 28))
        plt.savefig("4-" + str(num_4) + ".png")
        plt.clf()
        num_4 += 1
    if predicted[i] == 5:
        plt.imshow(x_train[i].reshape(28, 28))
        plt.savefig("5-" + str(num_5) + ".png")
        plt.clf()
        num_5 += 1
    if predicted[i] == 6:
        plt.imshow(x_train[i].reshape(28, 28))
        plt.savefig("6-" + str(num_6) + ".png")
        plt.clf()
        num_6 += 1



com_len = [len(comp) for comp in components]
print(com_len)

print(components[0][0:50])
print(components[1][0:50])
print(components[2][0:50])
print(components[3][0:50])
print(components[4][0:50])
print(components[5][0:50])
print(components[6][0:50])



# image_index = 7777 # You may select anything up to 60,000
# print(y_train[image_index]) # The label is 8
# plt.imshow(x_train[image_index], cmap='Greys')
# plt.show()

