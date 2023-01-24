import numpy as np
import math


def multi(W1, W2, X1):
    D = [[1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1]]
    alpha = 0.9
    N = 10

    for k in range(0, N):
        x = np.reshape(X1[k], 100)
        d = (D[k])
        d = np.transpose(d)
        v1 = np.matmul(W1, x)
        if k == 0:
            y1 = np.zeros((len(v1)))
            error = np.zeros(N)
        for ii in range(0, (len(v1))):
                y1[ii] = sigmoid(v1[ii])

        v = np.matmul(W2, y1)
        y = softmax(v)
        e = np.subtract(d, y)
        delta = e
        error[k] = np.mean(abs(e))
        e1 = np.matmul(np.transpose(W2), delta)
        y1_a = np.subtract(1, y1)
        delta1a = np.multiply(y1, y1_a)
        delta1 = np.multiply(delta1a, e1)
        transx = x.reshape(1, len(x))
        delta1 = delta1.reshape(len(delta1), 1)
        dW1 = np.matmul(delta1, transx)
        dW1a = np.multiply(dW1, alpha)
        W1 = np.add(W1, dW1a)
        transy1 = y1.reshape(1, len(y1))
        delta = delta.reshape(len(delta), 1)
        dW2 = np.matmul(delta, transy1)
        dW2a = np.multiply(alpha, dW2)
        W2 = np.add(W2, dW2a)
    e = np.mean(error)
    return W1, W2, e


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x):
    return np.exp(x)/sum(np.exp(x))
