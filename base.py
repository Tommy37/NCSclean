import numpy as np

from function import TestEnv


class NCNESTM(object):
    def __init__(self, **kwargs):
        self.distribution_size = kwargs['lam']
        self.pop_size = kwargs['mu']
        self.dimension = kwargs['D']
        self.fuc_id = kwargs['function_id']

        self.means_list, self.sigmas_list = self.initDistributions()
        self.rg = np.random.default_rng(12345)

        self.env = TestEnv(self.dimension, self.fuc_id)

    def reward(self, x_list):
        return np.array([self.env.evaluate(x) for x in x_list])

    def initDistributions(self):
        means_list = np.zeros([self.distribution_size, self.dimension])
        sigmas_list = np.zeros([self.distribution_size, self.dimension, self.dimension])

        for i in range(self.distribution_size):
            means_list[i] = np.ones(self.dimension)
            diag = np.ones(self.dimension)
            sigmas_list[i] = np.diag(diag)

        return means_list, sigmas_list

    def f_m_grad(self, X, mean, cov, rewards):
        """
        Line 14 of the algorithm.
        :param X: pop_size * d array, offsprings
        :param mean: d * 1 array
        :param cov: d * 1 array
        :param rewards: pop_size * 1 array
        :return: The gradient of f with respect to mean.
        """
        cov_inv = 1 / cov

        # Both approaches are OK.
        grad = np.zeros_like(mean, dtype=float)
        X_dif = X - mean
        for i in range(self.pop_size):
            grad += cov_inv * X_dif[i] * rewards[i]
        grad /= self.pop_size
        return grad
        # return cov_inv * np.dot((X - mean).T, rewards) / self.pop_size

    def f_cov_grad(self, X, mean, cov, rewards):
        """
        Line 15 of the algorithm.
        :param X: pop_size * d array, offsprings
        :param mean: d * 1 array
        :param cov: d * 1 array
        :param rewards: pop_size * 1 array
        :return: The gradient of f with respect to sigma.
        """

        cov_inv = 1 / cov
        grad = np.zeros_like(mean, dtype=float)
        X_dif = X - mean
        for i in range(self.pop_size):
            grad += (cov_inv * X_dif[i] * X_dif[i] * cov_inv - cov_inv) * rewards[i]
        grad /= 2 * self.pop_size
        return grad

    def fisher_m(self, X, mean, cov):
        """
        Line 18 of the algorithm.
        :param X: pop_size * d array, offsprings
        :param mean:
        :param cov:
        :return:
        """
        cov_inv = 1 / cov
        fisher = np.zeros_like(mean, dtype=float)
        X_dif = X - mean
        for i in range(self.pop_size):
            fisher += cov_inv * X_dif[i] * X_dif[i] * cov_inv
        fisher /= self.pop_size
        return fisher

    def fisher_sigma(self, X, mean, cov):
        """
        Line 19 of the algorithm.
        :param X: pop_size * d array, offsprings
        :param mean:
        :param cov:
        :return:
        """
        cov_inv = 1 / cov
        fisher = np.zeros_like(mean, dtype=float)
        X_dif = X - mean
        for i in range(self.pop_size):
            y = cov_inv * X_dif[i] * X_dif[i] * cov_inv - cov_inv
            fisher += y * y
        fisher /= 4 * self.pop_size
        return fisher
