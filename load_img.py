import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cbook as cbook

# Converts training images into binary matrices (1 = black, 0 = white)
def load_training_imgs():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "Images")

    training_matrix = np.zeros((10,10,10))

    training_matrix[0] = load_image(image_path + "\\1.png")

    training_matrix[1] = load_image(image_path + "\\1_1.png")

    training_matrix[2] = load_image(image_path + '\\2.png')

    training_matrix[3] = load_image(image_path + '\\2_1.png')

    training_matrix[4] = load_image(image_path + '\\3.png')

    training_matrix[5] = load_image(image_path + '\\3_1.png')

    training_matrix[6] = load_image(image_path + '\\4.png')

    training_matrix[7] = load_image(image_path + '\\4_1.png')

    training_matrix[8] = load_image(image_path + '\\5.png')

    training_matrix[9] = load_image(image_path + '\\5_1.png')

    return training_matrix;

def load_inference_imgs():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "Images")

    inference_matrix = np.zeros((5,10,10))

    inference_matrix[0] = load_image(image_path + "\\1_2.png")

    inference_matrix[1] = load_image(image_path + "\\2_2.png")

    inference_matrix[2] = load_image(image_path + '\\3_2.png')

    inference_matrix[3] = load_image(image_path + '\\4_2.png')

    inference_matrix[4] = load_image(image_path + '\\5_2.png')

    return inference_matrix;

def load_image(image_path):
    with cbook.get_sample_data(image_path) as image_file:
        image = plt.imread(image_file)
    pixels_matrix = invert_image_colors(image)
    return pixels_matrix

def invert_image_colors(image):
    image_shape = np.shape(image)
    inverted_image = np.zeros((image_shape[0], image_shape[1]))
    for ii in range(0, image_shape[0]):
        for iii in range(0, image_shape[1]):
            if(image[ii, iii, 0] == 1):
                inverted_image[ii, iii] = 0
            else:
                inverted_image[ii, iii] = 1
    return inverted_image;

