import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cbook as cbook

def load_img():
    a = np.zeros((10,10,10))
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\1.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[0] = invert(alpha,image)
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\1_1.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[1] = invert(alpha,image)
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\2.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[2] = invert(alpha,image)
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\2_1.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[3] = invert(alpha,image)
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\3.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[4] = invert(alpha,image)
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\3_1.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[5] = invert(alpha,image)
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\4.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[6] = invert(alpha,image)
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\4_1.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[7] = invert(alpha,image)
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\5.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[8] = invert(alpha,image)
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\5_1.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[9] = invert(alpha,image)
    return a;
def load_img_if():
    a = np.zeros((5,10,10))
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\1_2.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[0] = invert(alpha,image)
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\2_2.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[1] = invert(alpha,image)
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\3_2.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[2] = invert(alpha,image)
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\4_2.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[3] = invert(alpha,image)
    with cbook.get_sample_data('C:\\Users\\DELL\\Desktop\\imagens_IA\\5_2.png') as image_file:
        image = plt.imread(image_file)
    alpha = np.shape(image)
    a[4] = invert(alpha,image)
    return a;

def invert(alpha,X):
    a = np.zeros((alpha[0], alpha[1]))
    for ii in range(0, alpha[0]):
        for iii in range(0, alpha[1]):
            if (X[ii, iii, 0] == 1):
                a[ii, iii] = 0
            else:
                a[ii, iii] = 1
    return a;

