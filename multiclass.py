import numpy as np
import math

def multi(weights_1, weights_2, training_imgs):
    expected_outputs = [[1, 0, 0, 0, 0],
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
    n_training_imgs = len(training_imgs)
    y1 = np.zeros((len(weights_1)))
    error_array = np.zeros(n_training_imgs)
    height_pxls = np.size(training_imgs[0],0)
    width_pxls = np.size(training_imgs[0],1)
    n_pixels = height_pxls * width_pxls
    for img_index in range(0, n_training_imgs):
        training_img = np.reshape(training_imgs[img_index], n_pixels)
        expected_output = (expected_outputs[img_index])
        expected_output = np.transpose(expected_output)
        v1 = np.matmul(weights_1, training_img)
        
        for ii in range(0, (len(v1))):
            y1[ii] = sigmoid(v1[ii])

        v = np.matmul(weights_2, y1)
        y = softmax(v)
        error = np.subtract(expected_output, y)
        delta = error
        error_array[img_index] = np.mean(abs(error))
        e1 = np.matmul(np.transpose(weights_2), delta)
        y1_a = np.subtract(1, y1)
        delta1a = np.multiply(y1, y1_a)
        delta1 = np.multiply(delta1a, e1)
        transx = training_img.reshape(1, len(training_img))
        delta1 = delta1.reshape(len(delta1), 1)
        dW1 = np.matmul(delta1, transx)
        dW1a = np.multiply(dW1, alpha)
        weights_1 = np.add(weights_1, dW1a)
        transy1 = y1.reshape(1, len(y1))
        delta = delta.reshape(len(delta), 1)
        dW2 = np.matmul(delta, transy1)
        dW2a = np.multiply(alpha, dW2)
        weights_2 = np.add(weights_2, dW2a)
    error = np.mean(error_array)
    return weights_1, weights_2, error


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x):
    return np.exp(x)/sum(np.exp(x))
