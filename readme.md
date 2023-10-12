# Readme

本代码为 NCS-C 、 NCNES 算法的 python 版代码。

1. 其中评估部分提供了测试函数的 API 接口(function.py)，适应度函数定义在 `function.py` 的 `get_fitness()`，当前采用的是大规模函数Benchmark[1] 。

2. 优化的参数为 `self.params`，其变量规模在命令行参数D修改。


## Usage

NCS算法输入参数：
    --run_name, -r    本次运行的别名，保存日志文件时会用到这个名字
    --function_id, -f CEC测试函数的id，取值范围[1,2,3,4,5,6]
    --dimension, -d   CEC测试函数的维度
    --epoch,          NCS算法中更新高斯噪声标准差周期，一般取值为5的倍数
    --sigma0,         NCS算法中高斯噪声标准差的初始值
    --rvalue          NCS算法中更新高斯噪声标准差的系数

NCNES算法输入参数：
    --run_name, -r    本次运行的别名，保存日志文件时会用到这个名字
    --function_id, -f CEC测试函数的id，取值范围[1,2,3,4,5,6]
    --dimension, -d   CEC测试函数的维度
    --sigma0          NCNES算法中高斯噪声标准差的初始值
    --rvalue          NCNES算法中更新高斯噪声标准差的系数
    --lam             NCNES算法中种群数
    --mu              NCNES算法中种群中个体数
    --phi             NCNES算法中负相关系数
    --lr_sigma        NCNES算法中sigma梯度更新的学习率
    --lr_mean         NCNES算法中mean梯度更新的学习率

command:
```bash
./script/ncs.sh
./script/ncnes.sh
```

log

./logs_mpi/function1/NCS/...../log.txt