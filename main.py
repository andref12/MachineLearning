import numpy as np
import math
import time
import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt
from multiclass import multi, softmax, sigmoid
from load_img import load_training_imgs, load_inference_imgs

start_time = time.time()

#Plot training images
training_imgs = load_training_imgs()
f, axarr = plt.subplots(2, 5)
axarr[0, 0].imshow(training_imgs[0])
axarr[0, 1].imshow(training_imgs[1])
axarr[0, 2].imshow(training_imgs[2])
axarr[0, 3].imshow(training_imgs[3])
axarr[0, 4].imshow(training_imgs[4])
axarr[1, 0].imshow(training_imgs[5])
axarr[1, 1].imshow(training_imgs[6])
axarr[1, 2].imshow(training_imgs[7])
axarr[1, 3].imshow(training_imgs[8])
axarr[1, 4].imshow(training_imgs[9])

#Plot inference images
inference_imgs = load_inference_imgs()
fig, ax = plt.subplots(1, 5)
ax[0].imshow(inference_imgs[0])
ax[1].imshow(inference_imgs[1])
ax[2].imshow(inference_imgs[2])
ax[3].imshow(inference_imgs[3])
ax[4].imshow(inference_imgs[4])

#Training
n_epochs = 7
epochs_list = list(range(1,n_epochs+1))
#Initializing Network with Random Weights
n_pixels = 100
n_hidden_layer_nodes = 50
n_possible_outputs = 5
rng = np.random.default_rng(12345)
weights_1 = rng.random(size=(n_hidden_layer_nodes, n_pixels))
weights_1 = 2*weights_1 - 1
weights_2 = rng.random(size=(n_possible_outputs, n_hidden_layer_nodes))
weights_2 = 2*weights_2 - 1
med_err = np.zeros(len(epochs_list))
for r in range(1, len(epochs_list)+1):
    epoch = math.factorial(epochs_list[r-1])
    for ii in range(0, epoch):
        weights_1, weights_2, er = multi(weights_1, weights_2, training_imgs)
    sss = abs(er)
    med_err[r-1] = np.mean(sss)

eph = np.zeros(len(epochs_list))
for iii in range(1, len(epochs_list)+1):
    eph[iii-1] = math.factorial(epochs_list[iii-1])

figure, ax1 = plt.subplots()
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylabel('Error Rate')
ax1.set_xlabel('Epoch')
ax1.plot(eph, med_err)

#Inference
N_1 = [1, 2, 3, 4, 5]

def inference(weights_1, weights_2, inference_img):
    x_1 = np.reshape(inference_img, 100)
    v_11 = np.matmul(weights_1, x_1)
    y1 = np.zeros((len(v_11)))
    for ii in range(0, (len(v_11))):
        y1[ii] = sigmoid(v_11[ii])
    vv = np.matmul(weights_2, y1)
    yy = softmax(vv)
    return yy

inference_result_img_1 = inference(weights_1, weights_2, inference_imgs[0])
inference_result_img_2 = inference(weights_1, weights_2, inference_imgs[1])
inference_result_img_3 = inference(weights_1, weights_2, inference_imgs[2])
inference_result_img_4 = inference(weights_1, weights_2, inference_imgs[3])
inference_result_img_5 = inference(weights_1, weights_2, inference_imgs[4])

fi, ax2 = plt.subplots(1, 5)
ax2[0].bar(N_1, inference_result_img_1)
ax2[1].bar(N_1, inference_result_img_2)
ax2[2].bar(N_1, inference_result_img_3)
ax2[3].bar(N_1, inference_result_img_4)
ax2[4].bar(N_1, inference_result_img_5)

inference_result = np.concatenate((inference_result_img_1, inference_result_img_2, inference_result_img_3, inference_result_img_4, inference_result_img_5), axis=0)
inference_result = np.reshape(inference_result, (5, 5))
figura, a = plt.subplots()
df_cm = pd.DataFrame(inference_result, range(5), range(5))
a.set_title('Confusion Matrix')
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 25})

print("time elapsed: {:.2f}s".format(time.time() - start_time))
plt.show()
