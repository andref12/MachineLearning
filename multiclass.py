import numpy as np
import math

def multi(weights_1, weights_2, training_imgs, img_labels):
    #Fixed variables
    inertia = 0.9
    height_pxls = np.size(training_imgs[0],0)
    width_pxls = np.size(training_imgs[0],1)
    n_pixels = height_pxls * width_pxls
    n_possible_outputs = np.size(weights_2,0)
    n_training_imgs = len(training_imgs)
    #Initialize array with zeros
    input_layer_output_sigmoid = np.zeros((len(weights_1)))
    error_array = np.zeros(n_training_imgs)
    for img_index in range(0, n_training_imgs):
        training_img = np.reshape(training_imgs[img_index], n_pixels)
        expected_output_vector = expected_output(img_labels[img_index],n_possible_outputs)
        input_layer_output = np.matmul(weights_1, training_img)
        for ii in range(0, (len(input_layer_output))):
            input_layer_output_sigmoid[ii] = sigmoid(input_layer_output[ii])
        network_output = np.matmul(weights_2, input_layer_output_sigmoid)
        network_output_softmax = softmax(network_output)
        error = np.subtract(expected_output_vector, network_output_softmax)
        error_array[img_index] = np.mean(abs(error))
        #Backpropagation
        input_layer_error = np.matmul(np.transpose(weights_2), error)
        inverted_layer_output = np.subtract(1, input_layer_output_sigmoid)
        delta1a = np.multiply(input_layer_output_sigmoid, inverted_layer_output)
        delta1 = np.multiply(delta1a, input_layer_error)
        transx = training_img.reshape(1, len(training_img))
        delta1 = delta1.reshape(len(delta1), 1)
        dW1 = np.matmul(delta1, transx)
        dW1a = np.multiply(dW1, inertia)
        weights_1 = np.add(weights_1, dW1a)
        
        input_layer_out = input_layer_output_sigmoid.reshape(1, len(input_layer_output_sigmoid))
        error = error.reshape(len(error), 1)
        weights_2_correction = np.matmul(error, input_layer_out)
        weights_2_correction = np.multiply(inertia, weights_2_correction)
        weights_2 = np.add(weights_2, weights_2_correction)

    error = np.mean(error_array)
    return weights_1, weights_2, error

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def expected_output(img_label,n_possible_outputs):
    expected_output_vector = np.zeros((n_possible_outputs))
    expected_output_vector[img_label] = 1
    return expected_output_vector

