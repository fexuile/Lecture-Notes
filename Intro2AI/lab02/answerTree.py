import numpy as np
from copy import deepcopy
from typing import List, Callable

EPS = 1e-6

# 超参数，分别为树的最大深度、熵的阈值、信息增益函数
# TODO: You can change or add the hyperparameters here
hyperparams = {"depth":40, "purity_bound":0.1, "gainfunc":"gainratio"}

def entropy(Y: np.ndarray):
    n = Y.shape
    val, num = np.unique(Y,return_counts=True)
    return num.dot(np.log(num/n))

def gain(X: np.ndarray, Y: np.ndarray, idx: int):
    feat = X[:, idx]
    ufeat, featcnt = np.unique(feat, return_counts=True)
    featp = featcnt / feat.shape[0]

    ret = entropy(Y)
    for i in range(0,ufeat.shape[0]):
        ret -= featp[i] * entropy(Y[feat==ufeat[i]])
    return ret


def gainratio(X: np.ndarray, Y: np.ndarray, idx: int):
    ret = gain(X, Y, idx) / (entropy(X[:, idx]) + EPS)
    return ret


def giniD(Y: np.ndarray):
    u, cnt = np.unique(Y, return_counts=True)
    p = cnt / Y.shape[0]
    return 1 - np.sum(np.multiply(p, p))


def negginiDA(X: np.ndarray, Y: np.ndarray, idx: int):
    feat = X[:, idx]
    ufeat, featcnt = np.unique(feat, return_counts=True)
    featp = featcnt / feat.shape[0]
    ret = 0
    for i, u in enumerate(ufeat):
        mask = (feat == u)
        ret -= featp[i] * giniD(Y[mask])
    ret += giniD(Y)  # 调整为正值，便于比较
    return ret


class Node:
    def __init__(self): 
        self.children = {}
        self.featidx: int = None
        self.label: int = None

    def isLeaf(self):
        return len(self.children) == 0


def buildTree(X: np.ndarray, Y: np.ndarray, unused: List[int], depth: int, purity_bound: float, gainfunc: Callable, prefixstr=""):
    X = X > 0
    root = Node()
    u, ucnt = np.unique(Y, return_counts=True)
    root.label = u[np.argmax(ucnt)]
    #print(prefixstr, f"label {root.label} numbers {u} count {ucnt}") 
    purity = np.mean(Y!=root.label)
    if(depth == 0 or purity <= purity_bound):
        return root
    gains = [gainfunc(X, Y, i) for i in unused]
    idx = np.argmax(gains)
    root.featidx = unused[idx]
    unused = deepcopy(unused)
    unused.pop(idx)
    feat = X[:, root.featidx]
    ufeat = np.unique(feat)
    for x in ufeat:
        root.children[x]=buildTree(X[feat==x],Y[feat==x],unused, depth-1,purity_bound,gainfunc,prefixstr)
    return root


def inferTree(root: Node, x: np.ndarray):
    x = x > 0
    if root.isLeaf():
        return root.label
    child = root.children.get(x[root.featidx], None)
    return root.label if child is None else inferTree(child, x)

