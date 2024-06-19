import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *

# 超参数
# TODO: You can change the hyperparameters here
lr = 0.855   # 学习率
wd1 = 0.0007      # L1正则化
wd2 = 0.0005  # L2正则化
batchsize = 128

def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    # TODO: YOUR CODE HERE
    input_size = mnist.num_feat
    hidden1_size = 512
    hidden2_size = 256
    output_size = mnist.num_class
    nodes = [StdScaler(mnist.mean_X,mnist.std_X), 
             Linear(input_size, hidden1_size), 
            #  Dropout(),
             BatchNorm(hidden1_size),
             relu(),
             Linear(hidden1_size, hidden2_size),
            #  Dropout(),
             BatchNorm(hidden2_size),
             relu(),
             Linear(hidden2_size, output_size),
             LogSoftmax(), 
             NLLLoss(Y)
            # Softmax(),
            # CrossEntropyLoss(Y)
             ]
    graph=Graph(nodes)
    return graph
