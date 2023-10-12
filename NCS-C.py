
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   NCS-C
@Time    :   2020/06/27 16:41:31
@Describtion:  NCS-C 算法， 无强化学习
'''

import time
import pickle
import click
import os
import numpy as np

from mpi4py import MPI

from logger import Logger
from function import test_func_bound, TestEnv


class NCSAlgo(object):
    def __init__(self, **kwargs):
        '''算法类

        重要成员变量说明

            param     父代的参数（均值）
            param_all 所有父代的参数
            sigma     父代个体的协方差
            sigma_all 所有父代个体的协方差

            param_new 子代的参数

            BestParam_t     每个线程中的最优个体参数（分布均值）
            BestParam_t_all 所有线程中的最优个体参数集合
            BestScore_t     每个线程中的最优个体得分
            BestScore_t_all 所有线程中的最优个体得分集合

            BESTSCORE       所有线程中的最优个体得分
            BESTParam       所有线程中的最优个体参数

            reward_father  所有线程的父代适应度集合
            reward_child   所有线程的子代适应度集合
        '''
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.cpus = self.comm.Get_size()
        # 超参数设置
        self.args = kwargs
        self.lam = self.cpus - 1
        self.epoch = self.args['epoch']
        self.sigma0 = self.args['sigma0']
        self.r = self.args['r']
        self.updateCount = 0

        self.randomSeed = np.random.randint(100000)
        self.rs = np.random.RandomState(self.randomSeed)
        self.rs_rank = np.random.RandomState(self.rank+self.randomSeed)
        self.logger = Logger(self.logPath())

        # 同步不同线程的参数到param_all变量中
        bound = test_func_bound[self.args['function_id']]
        self.sigma0 = (bound[1] - bound[0]) / self.lam
        self.env = TestEnv(self.args['D'], self.args['function_id'])
        self.iter_max = self.args['D']
        self.param = self.rs_rank.uniform(bound[0], bound[1], self.args['D'])
        self.n = len(self.param)
        self.param_all = np.empty((self.cpus, self.n))
        self.comm.Allgather([self.param, MPI.DOUBLE], [self.param_all, MPI.DOUBLE])

        self.param_new = np.zeros(self.n)
        self.sigma = np.ones(self.n) * self.sigma0
        self.sigma_all = np.ones((self.cpus, self.n))
        self.BestParam_t = self.param.copy()
        self.BestParam_t_all = self.param_all.copy()
        self.BestScore_t = np.zeros(1)
        self.BestScore_t_all = np.zeros((self.cpus, 1))
        # BEST in all threads
        self.BESTSCORE = 0
        self.BESTSCORE_id = 0
        self.BESTParam = np.empty(self.n)

        self.logBasic()
        self.firstEvaluation()
        self.reward_child = None

        self.log_retest = {
            'iteration':[],
            'performance':[]
        }

    def firstEvaluation(self):
        """初始化种群后对个体进行评估
        """
        fit = self.evaluate(self.param)
        results = np.empty((self.cpus, 2))
        self.comm.Allgather([fit, MPI.DOUBLE], [results, MPI.DOUBLE])
        self.reward_father = results[:,:1].flatten()
        self.BestScore_t[0] = fit
        self.comm.Allgather([self.BestScore_t, MPI.DOUBLE], [self.BestScore_t_all, MPI.DOUBLE])
        self.udpateBEST()

    def udpateBEST(self):
        """依据每个线程中的最优个体来得到所有线程中的最优个体
        """
        self.comm.Allgather(
            [self.BestParam_t, MPI.DOUBLE], [self.BestParam_t_all, MPI.DOUBLE])
        self.comm.Allgather(
            [self.BestScore_t, MPI.DOUBLE], [self.BestScore_t_all, MPI.DOUBLE])

        self.BESTSCORE = np.min(self.BestScore_t_all.flatten()[1:])
        self.logger.log("updateBEST %e" % self.BESTSCORE)
        self.BESTSCORE_id = np.argmin(self.BestScore_t_all.flatten()[1:])+1
        self.BESTParam = self.BestParam_t_all[self.BESTSCORE_id].copy()

    def updateBest_t(self, score, param):
        """更新当前线程的最优个体得分与参数
        """
        if score < self.BestScore_t[0]:
            self.BestScore_t[0] = score
            self.BestParam_t = param.copy()

    def evaluate(self,parameters):
        """评估的API接口"""
        f = self.env.evaluate(parameters)
        return f

    def calLlambda(self, gen):
        """计算 llambda的值，这里采用llambda表示算法中的lambda，因为lambda在python中是一个关键字
        """
        self.llambda = self.rs.rand() * (0.1-0.1*gen/self.iter_max) + 1.0

    def logPath(self):
        """返回日志的路径
        """
        return "logs_mpi/function%s/NCS/lam%s/%s" %(self.args['function_id'], self.lam, self.args['run_name'])

    def run(self):
        """算法类的运行函数，即主循环
        """
        self.start_time = time.time()
        self.iteration = 1

        while self.iteration <= self.iter_max:
            iter_start_time = time.time()
            self.calLlambda(self.iteration)
            if self.iteration -1 == 0:
                # 每epoch代的第一代需要对更新次数统计变量self.updateCount进行清0
                self.updateCount = np.zeros(self.n, dtype=np.int32)

            # generate child and evaluate it
            self.generateAndEvalChild()
            self.udpateBEST()
            self.replaceFather()

            self.updateSigma()
            self.log(iter_start_time)
            self.iteration += 1
            self.retestBestFound(self.iteration)

    def retestBestFound(self, gen):
        """重新测试保存的所有线程的最优解 self.cpus * k 次
        """
        fitness = self.evaluate(self.BESTParam)
        fitness_all = self.syncOneValue(fitness)
        fitness_mean = np.mean(fitness_all)
        self.log_retest['iteration'].append(self.iteration)
        self.log_retest['performance'].append(fitness_mean)
        if self.rank == 0:
            if self.iteration % 10000 == 0:
                self.logger.save_parameters(self.BESTParam, self.iteration)

    def save_retest_log(self):
        """保存重新测试的日志
        """
        filepath = os.path.join(self.logPath(), 'retest_log.pickle')
        with open(filepath, 'wb') as f:
            pickle.dump(self.log_retest, f)

    def syncOneValue(self, v):
        """工具函数，用于同步每个线程的单个标量值到每个线程的向量中。

        对mpi的Allgather的简单封装

        参数：
            v：标量
        返回值：
            np.ndarray ： 大小为 cpus
        """
        v_t = np.array([v])
        v_all = np.zeros((self.cpus, 1))
        self.comm.Allgather([v_t, MPI.DOUBLE], [v_all, MPI.DOUBLE])
        return v_all.flatten()

    def generateAndEvalChild(self):
        """产生子代并进行评估

        非主线程都要产生一个子代，并进行评估，完成之
        后同步子代的适应度和消耗的训练帧数

        返回值：
            cost_steps ： 消耗的训练帧数
        """
        if self.rank != 0:
            # 生成子代
            self.param_new = self.param + self.rs_rank.normal(scale = self.sigma,size = self.n)
            #if self.rank == 1:
            #    self.logger.log("params-new %f"%np.mean(self.param_new))

            # 评估子代
            reward_child_t = self.evaluate(self.param_new)
            #if self.rank == 1:
            #    self.logger.log("reward_child_t%f "%reward_child_t)
            self.updateBest_t(reward_child_t, self.param_new)
        else:
            reward_child_t, reward_father_t = 0, 0

        # sync child reward
        self.reward_child = self.syncOneValue(reward_child_t)

    def replaceFather(self):
        """替换父代

        根据算法计算是否用子代替换父代，只有非主线程参与
        需要做的事情：
            1. 适应度归一化
            2. 计算相关性，并归一化
            3. 是否替换
            4. 是否需要更新父代个体的适应度值
            5. 同步父代的参数
        """
        if self.rank != 0:
            # refer to NCSCC pseudo code line 12
            father_corr = self.calCorr(self.param_all, self.param, self.sigma_all, self.sigma)
            child_corr = self.calCorr(self.param_all, self.param_new, self.sigma_all, self.sigma)

            # 每个线程计算自己的correlation和new correlation， 但是对于相关性和fitness都需要进行归一化
            child_corr = child_corr / (father_corr + child_corr + 1e-8)
            # 优化目标是到最小
            father_f = self.BESTSCORE - self.reward_father[self.rank]
            child_f = self.BESTSCORE - self.reward_child[self.rank]
            child_f = child_f / (child_f + father_f + 1e-8)

            #self.logger.log("child_f %f"%child_f)
            #self.logger.log("child_corr %f"%child_corr)
            if child_f / child_corr < self.llambda:
                # 抛弃旧的解，更换为新解
                self.param = self.param_new.copy()
                self.updateCount = self.updateCount + 1
                self.reward_father[self.rank] = self.reward_child[self.rank]
            reward_father_t = self.reward_father[self.rank]
        else:
            reward_father_t = 0
        self.comm.Allgather([self.param, MPI.DOUBLE], [self.param_all, MPI.DOUBLE])

    def updateSigma(self):
        """更新协方差，并同步
        """
        if self.rank != 0:
            if self.iteration % self.epoch == 0:
                self.sigma[self.updateCount/self.epoch<0.2] = self.sigma[self.updateCount/self.epoch<0.2] * self.r
                self.sigma[self.updateCount/self.epoch>0.2] = self.sigma[self.updateCount/self.epoch>0.2] / self.r
        self.comm.Allgather([self.sigma, MPI.DOUBLE], [self.sigma_all, MPI.DOUBLE])

    def log(self, iter_start_time):
        """日志函数
        """
        logger = self.logger
        if self.rank == 0:
            if self.iteration % 1000 == 0:
                logger.log("the best of iteration %d are %e" %(self.iteration, self.BESTSCORE))
                iteration_time = (time.time() - iter_start_time)
                time_elapsed = (time.time() - self.start_time)/60
                logger.log("Updatecount %d" %np.sum(self.updateCount))
                logger.log('IterationTime'.ljust(25) + '%f' % iteration_time)
                logger.log('TimeSinceStart'.ljust(25) + '%d' %time_elapsed)
                self.save_retest_log()

    def saveAndTestBest(self):
        """运行算法之后，对最好的个体进行测试，并保存结果
        """
        logger = self.logger
        msg = self.evaluate(self.BESTParam)

        if self.rank == 0:
            logger.log('Final'.ljust(25) + '%e' % msg)
            logger.save_parameters(self.BESTParam, self.iteration)
            time_elapsed = (time.time() - self.start_time)/60
            logger.log('TimeSinceStart'.ljust(25) + '%f' % time_elapsed)
            logger.log("random seed: %d" % self.randomSeed)
            self.save_retest_log()

    def logBasic(self):
        """基础信息的日志输出
        """
        logger = self.logger
        if self.rank == 0:
            logger.log("N:%d" % self.lam)
            logger.log("sigma0: %s" % str(self.sigma0))
            logger.log("epoch: %d" % self.epoch)
            logger.log("r:%f" % self.r)
            logger.log("Itermax:%d" % self.iter_max)

    @staticmethod
    def calBdistance(param1, param2, sigma1, sigma2):
        """计算分布之间的距离
        参数：
            param1(np.ndarray): 分布1的均值
            sigma1(np.ndarray): 分布1的协方差

            param2(np.ndarray): 分布2的均值
            sigma2(np.ndarray): 分布2的协方差

        返回值：
            分布之间的距离值
        """
        xi_xj = param1 - param2
        big_sigma1 = sigma1 * sigma1
        big_sigma2 = sigma2 * sigma2
        big_sigma = (big_sigma1 + big_sigma2) / 2
        small_value = 1e-8
        part1 = 1 / 8 * np.sum(xi_xj * xi_xj / (big_sigma + small_value))
        part2 = (
            np.sum(np.log(big_sigma + small_value))
            - 1 / 2 * np.sum(np.log(big_sigma1 + small_value))
            - 1 / 2 * np.sum(np.log(big_sigma2 + small_value))
        )
        return part1 + 1 / 2 * part2

    def calCorr(self, params_list, param, sigma_all, sigma):
        """计算分布param的相关性
        参数：
            n(int): the number of parameters

            param(np.ndarray): 当前分布的均值
            sigma(np.ndarray): 当前分布的协方差

            param_list(np.ndarray): 所有分布的均值
            sigma_all(np.ndarray): 所有分布的协方差

            rank(int): 当前线程的id
        返回值：
            这个分布的相关性
        """
        DBlist = []
        for i in range(len(params_list)):
            # i 是该进程在所有进程中序号
            if i != self.rank:
                param2 = params_list[i]
                sigma2 = sigma_all[i]
                DB = self.calBdistance(param, param2, sigma, sigma2)
                DBlist.append(DB)
        return np.min(DBlist)


'''
NCS算法的入口文件，用于处理输入参数。
输入参数：
    --run_name, -r    本次运行的别名，保存日志文件时会用到这个名字
    --function_id, -f CEC测试函数的id，取值范围[1,2,3,4,5,6]
    --dimension, -d   CEC测试函数的维度
    --epoch,          NCS算法中更新高斯噪声标准差周期，一般取值为5的倍数
    --sigma0,         NCS算法中高斯噪声标准差的初始值
    --rvalue          NCS算法中更新高斯噪声标准差的系数
'''

@click.command()
@click.option('--run_name', '-r', required=True, default = 'debug',type=click.STRING, help='Name of the run, used to create log folder name')
@click.option('--function_id', '-f', type=click.INT, help='function id for function optimization')
@click.option('--dimension', '-d', type=click.INT, help='the dimension of solution for function optimization')
@click.option('--epoch', type=click.INT, default=10, help='the number of epochs updating sigma')
@click.option('--sigma0', type=click.FLOAT, default=0.2, help='the intial value of sigma')
@click.option('--rvalue', type=click.FLOAT, default=0.8, help='sigma update parameter')
def main(run_name,  function_id, dimension, epoch, sigma0, rvalue):
    # 算法入口
    kwargs = {
        'epoch': epoch,
        'sigma0': sigma0,
        'run_name': run_name,
        'function_id': function_id,
        'D': dimension,
        'r': rvalue
    }
    algo = NCSAlgo(**kwargs)
    algo.run()
    # 最优解的最终测试以及保存日志
    algo.saveAndTestBest()


if __name__ == '__main__':
    main()

