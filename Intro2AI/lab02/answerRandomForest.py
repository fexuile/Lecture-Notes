from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 25     # 树的数量
ratio_data = 0.8   # 采样的数据比例
ratio_feat = 0.6 # 采样的特征比例
hyperparams = {"depth":23, "purity_bound":5e-2, "gainfunc":gainratio}


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    trees = list()
    n = X.shape[0]
    d = X.shape[1]
    for times in range(num_tree):
        unused = []
        samplelist = []
        for feat in range(d):
            randval = rand()
            if randval<=ratio_feat: unused.append(feat)
        for sample in range(n):
            randnum = rand()
            if randnum<=ratio_data: samplelist.append(sample)
        x = X[samplelist,:]
        y = Y[samplelist]
        trees.append(buildTree(x,y,unused,**hyperparams))
    return trees

def infertrees(trees, X):
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
