import numpy as np
from function import TestEnv

class SearchDistribution(object):
    def __init__(self, pop_size, dimension, env):
        self.dimension = dimension
        self.pop_size = pop_size
        self.mean = None
        self.cov = None
        self.sigma = None
        self.B = None
        self.initDistribution()
        self.env = env
        self.best_score = None

        self.eta = 0.5
        self.episodes = 50

        self.rg = np.random.default_rng()

    def initDistribution(self):
        self.mean = np.ones(self.dimension) * 100
        cov_diag = np.ones(self.dimension)
        self.cov = np.diag(cov_diag)
        A = decompose(self.cov)
        self.sigma = cal_sigma(A)
        self.B = A / self.sigma

    def update_best(self, rewards):
        for reward in rewards:
            if self.best_score is None or reward < self.best_score:
                self.best_score = reward


    def gen_offspring(self):
        s = self.rg.multivariate_normal(np.zeros(self.dimension), np.eye(self.dimension), self.pop_size)
        z = self.sigma * (self.B.T @ s.T).T + self.mean
        return s, z

    def cal_util(self, rewards):
        ranks = np.zeros_like(rewards)
        for r, idx in enumerate(np.argsort(rewards)[::-1]):
            ranks[idx] = r + 1
        mu = self.pop_size
        dividend = np.maximum(0., np.log(mu / 2 + 1) - np.log(ranks))
        divisor = dividend.sum()
        return (dividend / divisor) - (1 / mu)

    def test(self):
        for ep in range(self.episodes):
            s, z = self.gen_offspring()
            rewards = self.env.evaluate_mul(z)
            # self.update_best(rewards)
            best = None
            for reward in rewards:
                if best is None or reward < best:
                    best = reward
            print(best)
            u = self.cal_util(rewards)
            # print(s)
            # print(s.mean())
            # print(z)
            # print(z.mean())
            # print(u)

            grad_delta = np.zeros_like(self.mean)
            grad_M = np.zeros_like(self.cov)

            for i in range(self.pop_size):
                grad_delta += u[i] * s[i]

                grad_M += u[i] * (np.outer(s[i], s[i]) - np.eye(self.dimension))

            grad_delta /= self.pop_size
            grad_M /= self.pop_size
            grad_sigma = np.trace(grad_M) / self.dimension
            grad_B = grad_M - grad_sigma * np.eye(self.dimension)

            self.mean += self.eta * self.sigma * self.B @ grad_delta
            self.sigma *= np.exp(self.eta / 2 * grad_sigma)
            self.B *= np.exp(self.eta / 2 * grad_B)



def decompose(mat):
    evals, evecs = np.linalg.eig(mat)
    C = evecs @ (np.diag(evals ** 0.5))
    return C.T


def cal_sigma(mat):
    d = mat.shape[0]
    return np.power(abs(np.linalg.det(mat)), 1 / d)


if __name__ == '__main__':
    mu = 10
    d = 1
    gau = SearchDistribution(mu, d, TestEnv(d, 1))
    gau.test()