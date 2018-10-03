import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv
import random
import math

# initial_means is randomly picked data.
# initial covariance matrices are just identity matrix
# initial values of pi are just 1/num_cluster
def mixture_of_gaussian(data, num_iter, num_cluster):
    initial_cov = [[.1, 0.], [0., .1]]

    means = np.asarray([[0., 0.], [1., -1.], [-1., 1.]], dtype=np.float64)# random_pick(data, num_cluster)
    covariances = np.tile(initial_cov, [num_cluster, 1, 1])
    pis = np.full([num_cluster], 1/num_cluster)

    def Estep():
        resp = []
        for k in range(num_cluster):
            resp_temp = []
            for n in range(data.shape[0]):
                denom = 0
                for j in range(num_cluster):
                    denom += pis[j] * gaussian_density(data[n], means[j], covariances[j])
                resp_temp.append(pis[k] * gaussian_density(data[n], means[k], covariances[k]) / denom)
            resp.append(resp_temp)
        return resp

    def Mstep(resp):
        Nk = np.sum(resp, axis=1)
        for k in range(num_cluster):
            new_covar = 0
            for n in range(data.shape[0]):
                new_covar += resp[k][n] * np.matmul(np.transpose([data[n] - means[k]]), [data[n] - means[k]])
            new_covar /= Nk[k]
            covariances[k] = new_covar

            new_mean = 0
            for n in range(data.shape[0]):
                new_mean += resp[k][n] * data[n]
            new_mean /= Nk[k]
            means[k] = new_mean

            pis[k] = Nk[k] / data.shape[0]

    #This function only works at 2D.
    def plot():
        plt.clf()
        plt.scatter(np.transpose(data)[0], np.transpose(data)[1])
        for k in range(num_cluster):
            plot_x = []
            plot_y = []
            for nx in range(-250, 250):
                for ny in range(-250, 250):
                    x = nx / 125.
                    y = ny / 125.
                    vec = [x - means[k][0], y - means[k][1]]
                    if math.fabs(np.sum(np.matmul(np.matmul([vec], pinv(covariances[k])), np.transpose([vec]))) - 1) <= 0.05:
                        plot_x.append(x)
                        plot_y.append(y)
            plt.scatter(plot_x, plot_y, s=[3] * len(plot_x))
        plt.show()

    for iter in range(num_iter):
        if iter % 10 == 0:
            plot()
        responsibilities = Estep()
        Mstep(responsibilities)



def random_pick(data, num):
    randomly_picked_data = []
    for i in range(num):
        picked_index = random.randint(0, data.shape[0] - 1)
        picked_data = data[picked_index]
        randomly_picked_data.append(picked_data)
    return randomly_picked_data


def gaussian_density(x, mean, covariance):
    return 1 / (((2 * math.pi) ** (mean.shape[0] / 2))* (np.linalg.det(covariance) ** 0.5)) \
           * math.exp(-0.5 * np.matmul(np.matmul([x - mean], pinv(covariance)), np.transpose([x - mean])))


def main():
    points1 = np.random.multivariate_normal([0, 0], [[.02, 0.005], [0.005, .02]], 50)
    points2 = np.random.multivariate_normal([1, 1], [[.02, 0.005], [0.005, .02]], 50)
    points3 = np.random.multivariate_normal([-1, -1], [[.02, 0.005], [0.005, .02]], 50)

    data = np.concatenate((points1, points2))
    data = np.concatenate((data, points3))

    plt.figure()
    mixture_of_gaussian(data, 100, 3)


if __name__ == "__main__":
    main()