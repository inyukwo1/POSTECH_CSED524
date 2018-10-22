import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import time


def mixture_of_ppca(data, num_iter,
                        data_dim, hidden_dim, num_cluster):
    factors = np.random.normal(0, 1, [num_cluster, data_dim, hidden_dim])
    means = random_pick(data, num_cluster)
    variances = np.full([num_cluster], .7)
    pis = np.full([num_cluster], 1/num_cluster)

    def Estep():
        resp = []
        expect_sn = []
        expect_snsnt = []
        for k in range(num_cluster):
            eye = np.identity(data_dim, dtype=float)
            A = factors[k]
            AT = np.transpose(factors[k])
            AAT = np.matmul(A, AT)
            sigma_eye = variances[k] * eye
            Phi = np.matmul(AT, pinv(AAT + sigma_eye))
            resp_temp = []
            expect_sn_temp = []
            expect_snsnt_temp = []
            for n in range(data.shape[0]):
                denom = 0
                for j in range(num_cluster):
                    denom += pis[j] * multivariate_normal.pdf(data[n], mean=means[j], cov=np.matmul(factors[j], np.transpose(factors[j])) + variances[j] * eye)
                numer = pis[k] * multivariate_normal.pdf(data[n], mean=means[k], cov=AAT + sigma_eye)
                resp_temp.append(numer / denom)
                xn_muk = np.transpose([data[n] - means[k]])
                expect_sn_temp.append(np.matmul(Phi, xn_muk))
                expect_snsnt_temp.append(np.identity(hidden_dim, dtype=float) - np.matmul(Phi, A) + np.matmul(np.matmul(np.matmul(Phi, xn_muk), np.transpose(xn_muk)), np.transpose(Phi)))

            resp.append(resp_temp)
            expect_sn.append(expect_sn_temp)
            expect_snsnt.append(expect_snsnt_temp)
        return resp, expect_sn, expect_snsnt

    def Mstep(resp, expect_sn, expect_snsnt):
        Nk = np.sum(resp, axis=1)
        for k in range(num_cluster):
            # for A_k
            Ak_first = 0
            for n in range(data.shape[0]):
                xn_muk = np.transpose([data[n] - means[k]])
                Ak_first += resp[k][n] * np.matmul(xn_muk, np.transpose(expect_sn[k][n]))
            Ak_second = 0
            for n in range(data.shape[0]):
                Ak_second += resp[k][n] * expect_snsnt[k][n]
            factors[k] = np.matmul(Ak_first, pinv(Ak_second))

            # for mu_k
            mu_numer = 0
            for n in range(data.shape[0]):
                mu_numer += resp[k][n] * (data[n] - np.matmul(factors[k], expect_sn[k][n])[0])
            mu_numer /= Nk[k]

            #for sigma_k^2
            sigma_first = 0
            for n in range(data.shape[0]):
                xn_muk = np.transpose([data[n] - means[k]])
                sigma_first += resp[k][n] * np.matmul(xn_muk.transpose(), xn_muk)[0][0]
            sigma_second = 0
            for n in range(data.shape[0]):
                xn_muk = np.transpose([data[n] - means[k]])
                sigma_second += 2 * resp[k][n] * np.matmul(np.matmul(expect_sn[k][n].transpose(), factors[k].transpose()), xn_muk)[0][0]
            sigma_third = 0
            for n in range(data.shape[0]):
                sigma_third += resp[k][n] * np.trace(np.matmul(np.matmul(expect_snsnt[k][n], factors[k].transpose()), factors[k]))
            variances[k] = (sigma_first - sigma_second + sigma_third) / (data_dim * Nk[k])

            #for pi_k
            pis[k] = Nk[k] / data.shape[0]



    #This function only works at 2D.
    def plot(iter):
        start_time = time.time()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.transpose(data)[0], np.transpose(data)[1], np.transpose(data)[2])
        for k in range(num_cluster):
            x = factors[k][0][0]
            y = factors[k][1][0]
            z = factors[k][2][0]

            cosz = x / math.sqrt(x * x + y * y)
            sinz = y / math.sqrt(x * x + y * y)
            cosy = x / math.sqrt(x*x + z*z)
            siny = -z / math.sqrt(x*x + z*z)

            u = np.linspace(0, 2 * np.pi, 40)
            v = np.linspace(0, np.pi, 40)
            x = 1 * np.outer(np.cos(u), np.sin(v))
            y = max(0.1, variances[k]) * np.outer(np.sin(u), np.sin(v))
            z = max(0.1, variances[k]) * np.outer(np.ones(np.size(u)), np.cos(v))
            print(variances[k])

            for i in range(40):
                newx = cosz * x[i] - sinz * y[i]
                newy = sinz * x[i] + cosz * y[i]
                newz = z[i]

                newnewx = cosy * newx + siny * newz
                newnewy = newy
                newnewz = -siny * newx + cosy * newz
                x[i] = newnewx + means[k][0]
                y[i] = newnewy + means[k][1]
                z[i] = newnewz + means[k][2]

            ax.plot_surface(x, y, z)

        print("Drawing took  " + str(time.time() - start_time) + " seconds.")
        fig.savefig('iter'+str(iter)+".png")
        if iter %16 == 0:
            plt.show()
        plt.close(fig)

    for iter in range(num_iter):
        if iter % 1 == 0:
            plot(iter)

        start_time = time.time()
        resp, expect_sn, expect_snsnt = Estep()
        Mstep(resp, expect_sn, expect_snsnt)
        print("Iter " + str(iter) + " done, took " + str(time.time() - start_time) + " seconds.")


def random_pick(data, num):
    randomly_picked_data = []
    for i in range(num):
        picked_index = random.randint(0, data.shape[0] - 1)
        picked_data = data[picked_index]
        randomly_picked_data.append(picked_data)
    return randomly_picked_data


def spiral_synthetic_data():
    data = []
    step_size = 0.01
    data_num = 500
    radius = 1.

    current_theta = 0
    current_z = 0
    for i in range(data_num):
        current_z += step_size / 2
        current_theta += step_size * 4
        new_data = [radius * math.cos(current_theta), radius * math.sin(current_theta), current_z]
        new_data += np.random.normal(0, 0.1, 3)
        data.append(new_data)
    return np.asarray(data)


def my_synthetic_data():
    data = []

    for i in range(100):
        current_step = i / 50
        new_data = [-1 + current_step, -1 + current_step, -1 + current_step]
        new_data += np.random.normal(0, 0.1, 3)
        data.append(new_data)
    for i in range(100):
        current_step = i / 50
        new_data = [-1 + current_step, 1 - current_step, 1 - current_step]
        new_data += np.random.normal(0, 0.1, 3)
        data.append(new_data)
    for i in range(100):
        current_step = i / 50
        new_data = [-1 + current_step, -1 + current_step, 1 - current_step]
        new_data += np.random.normal(0, 0.1, 3)
        data.append(new_data)
    for i in range(100):
        current_step = i / 50
        new_data = [-1 + current_step, 1 - current_step, -1 + current_step]
        new_data += np.random.normal(0, 0.1, 3)
        data.append(new_data)
    return np.asarray(data)

def main():
    data = my_synthetic_data()
    mixture_of_ppca(data, 1000, 3, 1, 8)


if __name__ == "__main__":
    main()