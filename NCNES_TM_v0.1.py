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
from function import test_func_bound, TestEnv
import click


# comm = MPI.COMM_WORLD


class NCNESTM(object):
    def __init__(self, **kwargs):
        # self.rank = comm.Get_rank()
        # self.uni = self.rank
        # self.comm_list = np.zeros([8, 1])

        self.distribution_size = kwargs['lam']
        self.pop_size = kwargs['mu']
        self.phi = kwargs['phi']
        self.learning_rate_m = kwargs['lr_mean']
        self.learning_rate_cov = kwargs['lr_sigma']
        self.dimension = kwargs['D']
        self.fuc_id = kwargs['function_id']
        self.eposides = 500

        self.mean_list, self.cov_list = self.initDistributions()
        self.rg = np.random.default_rng()

        self.env = TestEnv(self.dimension, self.fuc_id)
        self.epsilon = 1e-8
        self.best_score = None

        print(f"""Start Running, 
lam: {self.distribution_size}
mu: {self.pop_size}
phi: {self.phi}
lrm: {self.learning_rate_m}
lrcov: {self.learning_rate_cov}
d: {self.dimension}
o: {self.env.o}
""")
        print(self.mean_list)
        print(self.cov_list)

    def reward(self, x_list):
        return np.array([self.env.evaluate(x) for x in x_list])

    def initDistributions(self):
        """
        Needs further development to generate a better initial distribution.
        :return:
        """
        mean_list = np.ones([self.distribution_size, self.dimension])
        cov_list = np.ones([self.distribution_size, self.dimension])
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
        for ep in range(self.eposides):
            mean_list = np.zeros_like(self.mean_list)
            cov_list = np.zeros_like(self.cov_list)
            for i in range(self.distribution_size):
                mean = self.mean_list[i]
                cov = self.cov_list[i]

                children = self.rg.multivariate_normal(mean, np.diag(cov), self.pop_size)
                rewards = evaluate(children, self.env)
                self.update_score(rewards)

                util = self.cal_util(rewards)

                f_m_grad = self.f_m_grad(children, mean, cov, util)
                f_cov_grad = self.f_cov_grad(children, mean, cov, util)
                # d_m_grad = self.d_m_grad(mean, cov, self.mean_list, self.cov_list)
                # d_cov_grad = self.d_cov_grad(mean, cov, self.mean_list, self.cov_list)
                fisher_m = self.fisher_m(children, mean, cov)
                fisher_cov = self.fisher_cov(children, mean, cov)
                print()
                print('fmg: ', f_m_grad)
                print('fcovg: ', f_cov_grad)
                print('fisher_m: ', fisher_m)
                print('fisher_cov: ', fisher_cov)
                print()

                mean += self.learning_rate_m * (1 / fisher_m) * (f_m_grad)  # + self.phi * d_m_grad)
                cov += self.learning_rate_cov * (1 / fisher_cov) * (f_cov_grad)  # + self.phi * d_cov_grad)

                mean_list[i] += mean
                cov_list[i] += cov

                print('mean', mean)
                print('cov', cov)
            self.mean_list = mean_list
            self.cov_list = cov_list

            # if ep % 5 == 0:
            print('Best score %d, ep%d' % (self.best_score, ep))

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


def main():
    model = NCNESTM()
    model.fit()

@click.command()
@click.option('--run_name', '-r', required=False, type=click.STRING,
              help='Name of the run, used to create log folder name')
@click.option('--function_id', '-f', type=click.INT, help='function id for function optimization')
@click.option('--dimension', '-d', type=click.INT, default=100,
              help='the dimension of solution for function optimization')
@click.option('--sigma0', type=click.FLOAT, default=2, help='the intial value of sigma')
@click.option('--rvalue', type=click.FLOAT, default=0.99, help='sigma update parameter')
@click.option('--lam', type=click.INT, default=5, help='population nums')
@click.option('--mu', type=click.INT, default=15, help='population size')
@click.option('--parallel', '-p', type=click.STRING, default='p', help='parallel mode')
@click.option('--phi', type=click.FLOAT, default=0.00001, help='negative correation factor')
@click.option('--lr_sigma', '-etac', type=click.FLOAT, default=0.2, help='sigma learning rate')
@click.option('--lr_mean', '-etam', type=click.FLOAT, default=0.1, help='mean learning rate')
def main(run_name, function_id, dimension, sigma0, rvalue, parallel, phi, lam, mu, lr_sigma, lr_mean):
    # 算法入口
    kwargs = {
        # 'sigma0': sigma0,
        # 'run_name': run_name,
        'function_id': function_id,
        'D': dimension,
        'r': rvalue,
        # 'H': 100,
        # 'L': -100,
        'parallel': parallel,
        'lr_sigma': lr_sigma,
        'lr_mean': lr_mean,
        'phi': phi,
        'lam': lam,
        'mu': mu
    }
    model = NCNESTM(**kwargs)
    model.fit()


if __name__ == '__main__':
    main()
