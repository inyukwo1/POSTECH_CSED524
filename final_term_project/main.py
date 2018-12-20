import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import csv

from sklearn import mixture

color_iter = itertools.cycle(['red'])


def plot_results(X, Y_, means, covariances, title):
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color="black")

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0] * 2, v[1] * 2, 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.3)
        splot.add_artist(ell)

    plt.xlim(0., 7.)
    plt.ylim(20., 100.)
    plt.title(title)


data = []
f = open('old_faithful.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
for line in rdr:
    x, y = line
    data.append([float(x), float(y)])
data = np.array(data, dtype=float)

# Fit a Dirichlet process Gaussian mixture using five components
dpgmm = mixture.BayesianGaussianMixture(n_components=100,
                                        covariance_type='full', random_state=12).fit(data)
plot_results(data, dpgmm.predict(data), dpgmm.means_, dpgmm.covariances_, '')

plt.show()

# filename = 'glove.6B.50d.txt'
# def loadGloVe(filename):
#     vocab = []
#     embd = []
#     file = open(filename,'r')
#     for line in file.readlines():
#         row = line.strip().split(' ')
#         vocab.append(row[0])
#         embd.append(row[1:])
#     print('Loaded GloVe!')
#     file.close()
#     return vocab,embd
# vocab,embd = loadGloVe(filename)
# print("loaded")
# vocab_size = len(vocab)
# print(vocab_size)
# embedding_dim = len(embd[0])
# embedding = np.asarray(embd)[0:10000]
# vocab = vocab[0:10000]
#
# print("start")
# print(datetime.datetime.now())
# dpgmm = mixture.BayesianGaussianMixture(n_components=100,
#                                         covariance_type='full', random_state=2).fit(embedding)
# print(datetime.datetime.now())
# print("fitted")
# predicted = dpgmm.predict(embedding)
# print("predicted")
# components = list([] for i in range(100))
#
# for i in range(len(vocab)):
#     components[predicted[i]].append(vocab[i])
#
# com_len = [len(comp) for comp in components]
# print(com_len)
#
# print(components[0][0:50])
# print(components[1][0:50])
# print(components[2][0:50])
# print(components[3][0:50])
# print(components[4][0:50])
# print(components[5][0:50])
# print(components[6][0:50])