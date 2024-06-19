import numpy as np

lr1 = 0.61 # 学习率
lr2 = 0.204
wd = 1e-5  # l2正则化项系数 

def predict(X, w, b):
    return np.dot(X, w) + b

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def mylog(x):
    if x>=15:
        return x+np.log(1+np.exp(-x))
    else:
        return np.log(1+np.exp(x))

def step(X, w, b, y):
    n = X.shape[0]

    haty = predict(X,w,b)
    tmp = np.zeros((n))
    for i in range(0,n): tmp[i] = mylog(-y[i]*haty[i])
    loss = np.mean(tmp)+wd*np.sum(w**2)

    dw = -np.dot(X.T, (1-sigmoid(y*haty))*y) / n + 2 * wd * w
    db = -np.mean(y*(1-sigmoid(y*haty)))

    if loss < 550: lr = lr2
    else: lr = lr1

    w -= lr * dw
    b -= lr * db

    return haty,loss,w,b
