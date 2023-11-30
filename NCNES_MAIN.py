# -*- encoding: utf-8 -*-
"""
NCNESTM
=======
File    :   NCNES.py

Time    :   2023/11/18 17:18:31

Author  :   Tang Qian


Description:  I briefly implements the algorithm in the paper, page 6. Note that the fisher matrix and
the covariance matrix are represented as a vector, which contains the diagonal elements of the supposed matrices.
In other words, we are just manipulating the diagonal elements of all the matrices, by seeing them as vectors also saves
times.
However, I just built a simple model, how to update the hyperparameters is the remaining task.

"""

from mpi4py import MPI
import numpy as np
import math
from function import test_func_bound, TestEnv
import click


# comm = MPI.COMM_WORLD


class NCNESTM(object):
    def __init__(self, **kwargs):
        # self.rank = comm.Get_rank()
        # self.uni = self.rank
        # self.comm_list = np.zeros([8, 1])

        self.distribution_size = 0 #kwargs['lam']
        self.pop_size = 0 #kwargs['mu']
        self.phi = 0 #kwargs['phi']
        self.learning_rate_m = 0 #kwargs['lr_mean']
        self.learning_rate_cov = 0 #kwargs['lr_sigma']
        self.dimension = kwargs['D']
        self.fuc_id = kwargs['function_id']
        self.L = test_func_bound[self.fuc_id][0]
        self.H = test_func_bound[self.fuc_id][1]

        self.iter_max = 10000 * self.dimension
        self.iter_cur = 0

        self.mean_list = None
        self.cov_list = None

        # self.mean_list, self.cov_list = self.initDistributions()
        self.rg = np.random.default_rng()

        self.env = TestEnv(self.dimension, self.fuc_id)
        self.epsilon = 1e-8
        self.best_score = None

        self.set_hyperparameters()
        self.display_info()

        self.flag = True

    def display_info(self):
        print(f'Lam: {self.distribution_size}')
        print(f'Mu: {self.pop_size}')
        print(f'Phi: {self.phi}')
        print(f"D: {self.dimension}")
        print(f'Learning rate m: {self.learning_rate_m}')
        print(f'Learning rate cov: {self.learning_rate_cov}')
        print()
        print(f'Mean: {self.mean_list}')
        print(f'Cov: {self.cov_list}')

    def set_hyperparameters(self):
        logd = np.log(self.dimension)
        self.distribution_size = math.ceil(logd)
        self.mean_list = np.zeros([self.distribution_size, self.dimension])
        self.cov_list = np.zeros([self.distribution_size, self.dimension])
        self.pop_size = 4 + math.floor(3 * logd)
        self.phi = 0.0001
        self.learning_rate_m = 1
        self.learning_rate_cov = (3 + logd) / (5 * np.sqrt(self.dimension))

        cov_init = np.ones(self.dimension) * ((self.H - self.L) / self.distribution_size)
        for i in range(self.distribution_size):
            self.mean_list[i] = self.rg.uniform(self.L, self.H, self.dimension)
        self.cov_list = self.cov_list + cov_init

    def check_pos(self):
        if not self.flag: return
        for cov in self.cov_list:
            for ele in cov:
                if ele <= 0:
                    self.flag = False
                    print('Not positive.')
                    print(self.cov_list)
                    return
    def reward(self, x_list):
        return self.env.evaluate_mul(x_list)

    def initDistributions(self):
        """
        Needs further development to generate a better initial distribution.
        :return:
        """
        mean_list = np.ones([self.distribution_size, self.dimension]) * 50
        cov_list = np.ones([self.distribution_size, self.dimension]) * 100
        # for i in range(self.distribution_size):
        #     means_list[i] = np.ones(self.dimension)
        #     diag = np.ones(self.dimension)
        #     sigmas_list[i] = np.diag(diag)
        return mean_list, cov_list

    def update_score(self, rewards):
        for reward in rewards:
            if self.best_score is None or reward < self.best_score:
                self.best_score = reward

    def fit(self):
        abcd = 0
        while self.iter_cur < self.iter_max:
            if abcd < 10:
                print(self.cov_list)
            abcd += 1
            lr_m, lr_cov = self.lr_decay()
            self.iter_cur += self.pop_size

            mean_list = np.zeros_like(self.mean_list)
            cov_list = np.zeros_like(self.cov_list)
            for i in range(self.distribution_size):
                mean = self.mean_list[i]
                cov = self.cov_list[i]


                children = self.rg.multivariate_normal(mean, np.diag(cov), self.pop_size)
                rewards = self.env.evaluate_mul(children)
                self.update_score(rewards)

                util = self.cal_util(rewards)

                f_m_grad = self.f_m_grad(children, mean, cov, util)
                f_cov_grad = self.f_cov_grad(children, mean, cov, util)
                d_m_grad = self.d_m_grad(mean, cov, self.mean_list, self.cov_list)
                d_cov_grad = self.d_cov_grad(mean, cov, self.mean_list, self.cov_list)
                fisher_m = self.fisher_m(children, mean, cov)
                fisher_cov = self.fisher_cov(children, mean, cov)
                # print()
                # print('Grad for mean: ', f_m_grad)
                # print('Grad for cov: ', f_cov_grad)
                # print('Fisher_m: ', fisher_m)
                # print('Fisher_cov: ', fisher_cov)
                # print()

                mean += lr_m * (1 / fisher_m) * (f_m_grad + self.phi * d_m_grad)
                cov += lr_cov * (1 / fisher_cov) * (f_cov_grad + self.phi * d_cov_grad)

                # mean += (1 / fisher_m) * (f_m_grad)  # + self.phi * d_m_grad)
                # cov += (1 / fisher_cov) * (f_cov_grad)  # + self.phi * d_cov_grad)

                mean_list[i] += mean
                cov_list[i] += cov

                # print('Mean: ', mean)
                # print('Cov: ', cov)
            self.mean_list = mean_list
            self.cov_list = cov_list
            self.check_pos()
            if (self.iter_cur / self.pop_size) % 50 == 0:
                print('Best score %.2f, ep%d' % (self.best_score, self.iter_cur))
                print()


    def cal_util(self, rewards):
        ranks = np.zeros_like(rewards)
        for r, idx in enumerate(np.argsort(rewards)[::-1]):
            ranks[idx] = r + 1
        mu = self.pop_size
        dividend = np.maximum(0., np.log(mu / 2 + 1) - np.log(ranks))
        divisor = dividend.sum()
        return (dividend / divisor) - (1 / mu)

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

    def d_m_grad(self, mean, cov, mean_list, cov_list):
        """
        Line 16 of the algorithm.
        :param mean: d * 1 array
        :param cov: d * 1 array
        :param mean_list: distribution_size * d matrix
        :param cov_list: distribution_size * d matrix
        :return:
        """
        grad = np.zeros_like(mean, dtype=float)
        for i in range(self.distribution_size):
            cov_add_inv = 2 / (cov + cov_list[i])
            mean_dif = mean - mean_list[i]
            grad += cov_add_inv * mean_dif
        grad /= 4
        return grad

    def d_cov_grad(self, mean, cov, mean_list, cov_list):
        """
        Line 17 of the algorithm.
        :param mean: d * 1 array
        :param cov: d * 1 array
        :param mean_list: distribution_size * d matrix
        :param cov_list: distribution_size * d matrix
        :return:
        """
        grad = np.zeros_like(mean, dtype=float)
        for i in range(self.distribution_size):
            cov_add_inv = 2 / (cov + cov_list[i])
            mean_dif = mean - mean_list[i]
            grad += cov_add_inv - (cov_add_inv * mean_dif * mean_dif * cov_add_inv) / 4 - (1 / cov)
        grad /= 4
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
        grad = np.zeros_like(mean, dtype=float)
        X_dif = X - mean
        for i in range(self.pop_size):
            grad += cov_inv * X_dif[i] * X_dif[i] * cov_inv
        grad /= self.pop_size
        return grad

    def fisher_cov(self, X, mean, cov):
        """
        Line 19 of the algorithm.
        :param X: pop_size * d array, offsprings
        :param mean:
        :param cov:
        :return:
        """
        cov_inv = 1 / cov
        grad = np.zeros_like(mean, dtype=float)
        X_dif = X - mean
        for i in range(self.pop_size):
            y = cov_inv * X_dif[i] * X_dif[i] * cov_inv - cov_inv
            grad += y * y
        grad /= 4 * self.pop_size
        return grad

    def lr_decay(self):
        factor = (np.e - np.exp(self.iter_cur / self.iter_max)) / (np.e - 1)
        lr_m = self.learning_rate_m * factor
        lr_cov = self.learning_rate_cov * factor
        return lr_m, lr_cov


def evaluate(X, env):
    """
    Give the rewards based on the input X.
    :param X: input_size * d array
    :param env: The env we want to use
    :return:
    """
    pop_size = X.shape[0]
    rewards = np.zeros(pop_size)
    for i in range(pop_size):
        rewards[i] += env.evaluate(X[i])
    return rewards


@click.command()
@click.option('--run_name', '-r', required=False, type=click.STRING,
              help='Name of the run, used to create log folder name')
@click.option('--function_id', '-f', type=click.INT, default=1,
              help='function id for function optimization')
@click.option('--dimension', '-d', type=click.INT, default=100,
              help='the dimension of solution for function optimization')
@click.option('--parallel', '-p', type=click.STRING, default='p', help='parallel mode')
def main(run_name, function_id, dimension, parallel):
    # 算法入口
    kwargs = {
        # 'run_name': run_name,
        'function_id': function_id,
        'D': dimension,
        'parallel': parallel,
    }
    model = NCNESTM(**kwargs)
    model.fit()


if __name__ == '__main__':
    main()
