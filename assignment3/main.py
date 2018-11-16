import matplotlib
matplotlib.use('TkAgg')
from scipy import special
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg.linalg import inv, det
import random
import math
import csv
from matplotlib.patches import Ellipse

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).
    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=0.2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def random_pick(data, num):
    randomly_picked_data = []
    for i in range(num):
        picked_index = random.randint(0, data.shape[0] - 1)
        picked_data = data[picked_index]
        randomly_picked_data.append(picked_data)
    return randomly_picked_data


def variational_MoG(data, num_iter, num_cluster):
    # initialize
    alpha_0s = np.asarray([0.1] * num_cluster)
    beta_0s = np.asarray([1.] * num_cluster)

    W_0s = np.asarray([[[.05, 0], [0, .05]]] * num_cluster)
    # for k in range(num_cluster):
    #     randnum = random.random() * 5 - 2.5
    #     W_0_invs.append([[random.random() * 10., randnum], [randnum, random.random() * 10.]])
    W_0s = np.asarray(W_0s)
    nu_0s = np.asarray([50.] * num_cluster)
    m_0s = np.asarray(random_pick(data, num_cluster))

    D = 2
    data_len = data.shape[0]

    alphas = alpha_0s.copy()
    betas = beta_0s.copy()
    Ws = W_0s.copy()
    nus = nu_0s.copy()
    ms = m_0s.copy()

    def variational_E():
        E_log_pi = special.digamma(alphas) - special.digamma(np.sum(alphas))
        E_log_Lambda = np.zeros((num_cluster,), dtype=float)
        for k in range(num_cluster):
            sum = 0.
            for i in range(D):
                sum += special.digamma((nus[k] - i) / 2)
            E_log_Lambda[k] = sum + D * math.log(2) + math.log(det(Ws[k]))
        E_comp = np.zeros((num_cluster, data_len), dtype=float)
        for k in range(num_cluster):
            for n in range(data_len):
                E_comp[k, n] = D / betas[k] + nus[k] * np.matmul(np.matmul([data[n] - ms[k]], Ws[k]), np.transpose([data[n] - ms[k]]))
        log_rhos = -0.5 * E_comp
        for k in range(num_cluster):
            for n in range(data_len):
                log_rhos[k, n] += E_log_pi[k] + 0.5 * E_log_Lambda[k] - D / 2 * math.log(2 * math.pi)
        rhos = np.exp(log_rhos)
        sum_rhos = np.sum(rhos, axis=0)
        resps = np.zeros((num_cluster, data_len), dtype=float)
        for k in range(num_cluster):
            for n in range(data_len):
                resps[k,n] = rhos[k, n] / sum_rhos[n]
        return resps

    def variational_M(resps):
        Ns = np.sum(resps, axis=1)
        xbar = np.zeros((num_cluster, D), dtype=float)
        for k in range(num_cluster):
            sum = 0
            for n in range(data_len):
                sum += resps[k, n] * data[n]
            xbar[k] = sum / Ns[k]
        Zs = np.zeros((num_cluster, D, D), dtype=float)
        for k in range(num_cluster):
            sum = np.zeros((D, D), dtype=float)
            for n in range(data_len):
                sum += resps[k, n] * np.matmul(np.transpose([data[n] - xbar[k]]), [data[n] - xbar[k]])
            Zs[k] = sum / Ns[k]
        for k in range(num_cluster):
            betas[k] = beta_0s[k] + Ns[k]
        for k in range(num_cluster):
            ms[k] = (beta_0s[k] * m_0s[k] + Ns[k] * xbar[k]) / betas[k]
        for k in range(num_cluster):
            nus[k] = nu_0s[k] + Ns[k]
        for k in range(num_cluster):
            Ws[k] = inv(inv(W_0s[k]) + Ns[k] * Zs[k] + (beta_0s[k] * Ns[k]) / (beta_0s[k] + Ns[k]) * np.matmul(np.transpose([xbar[k] - m_0s[k]]), [xbar[k] - m_0s[k]]))
        for k in range(num_cluster):
            alphas[k] = alpha_0s[k] + Ns[k]

    def plot():
        fig, ax = plt.subplots()
        ax.scatter(np.transpose(data)[0], np.transpose(data)[1])
        for k in range(num_cluster):
            alpha = alphas[k] / np.sum(alphas)
            # if (alpha < 0.1):
            #     continue
            plot_cov_ellipse(Ws[k] * nus[k], ms[k], ax=ax, color=(1, 0, 0, alpha))

    for iter in range(num_iter):
        print("iter " + str(iter))
        if iter % 5 == 0:
            plot()
        resps = variational_E()
        variational_M(resps)


def main():
    data = []
    f = open('old_faithful.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        x, y = line
        data.append([(float(x) - 3.5) / 2, (float(y) - 70) / 30])
    data = np.array(data, dtype=float)
    f.close()
    variational_MoG(np.asarray(data), 50, 10)


if __name__ == "__main__":
    main()