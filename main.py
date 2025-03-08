import numpy as np
import math
import time
import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt
from multiclass import multi, softmax, sigmoid
from teste import load_training_imgs, load_inference_imgs

start_time = time.time()

X = load_training_imgs()
f, axarr = plt.subplots(2, 5)
axarr[0, 0].imshow(X[0])
axarr[0, 1].imshow(X[1])
axarr[0, 2].imshow(X[2])
axarr[0, 3].imshow(X[3])
axarr[0, 4].imshow(X[4])
axarr[1, 0].imshow(X[5])
axarr[1, 1].imshow(X[6])
axarr[1, 2].imshow(X[7])
axarr[1, 3].imshow(X[8])
axarr[1, 4].imshow(X[9])

X0 = load_inference_imgs()
fig, ax = plt.subplots(1, 5)
ax[0].imshow(X0[0])
ax[1].imshow(X0[1])
ax[2].imshow(X0[2])
ax[3].imshow(X0[3])
ax[4].imshow(X0[4])

# treinamento
a = [1, 2, 3, 4, 5, 6, 7]
rng = np.random.default_rng(12345)
W1 = rng.random(size=(50, 100))
W1 = 2*W1 - 1
W2 = rng.random(size=(5, 50))
W2 = 2*W2 - 1
med_err = np.zeros(len(a))
for r in range(1, len(a)+1):
    epoch = math.factorial(a[r-1])
    for ii in range(0, epoch):
        W1, W2, er = multi(W1, W2, X)
    sss = abs(er)
    med_err[r-1] = np.mean(sss)

eph = np.zeros(len(a))
for iii in range(1, len(a)+1):
    eph[iii-1] = math.factorial(a[iii-1])

figure, ax1 = plt.subplots()
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylabel('Taxa de erro')
ax1.set_xlabel('Epoca')
ax1.plot(eph, med_err)

# inferencia

N_1 = [1, 2, 3, 4, 5]
x_1 = np.reshape(X0[0], 100)
v_11 = np.matmul(W1, x_1)
y1 = np.zeros((len(v_11)))
for ii in range(0, (len(v_11))):
    y1[ii] = sigmoid(v_11[ii])
vv = np.matmul(W2, y1)
yy = softmax(vv)

x_1 = np.reshape(X0[1], 100)
v_11 = np.matmul(W1, x_1)
for ii in range(0, (len(v_11))):
    y1[ii] = sigmoid(v_11[ii])
vv = np.matmul(W2, y1)
yy2 = softmax(vv)

x_1 = np.reshape(X0[2], 100)
v_11 = np.matmul(W1, x_1)
for ii in range(0, (len(v_11))):
    y1[ii] = sigmoid(v_11[ii])
vv = np.matmul(W2, y1)
yy3 = softmax(vv)

x_1 = np.reshape(X0[3], 100)
v_11 = np.matmul(W1, x_1)
for ii in range(0, (len(v_11))):
    y1[ii] = sigmoid(v_11[ii])
vv = np.matmul(W2, y1)
yy4 = softmax(vv)

x_1 = np.reshape(X0[4], 100)
v_11 = np.matmul(W1, x_1)
for ii in range(0, (len(v_11))):
    y1[ii] = sigmoid(v_11[ii])
vv = np.matmul(W2, y1)
yy5 = softmax(vv)

fi, ax2 = plt.subplots(1, 5)
ax2[0].bar(N_1, yy)
ax2[1].bar(N_1, yy2)
ax2[2].bar(N_1, yy3)
ax2[3].bar(N_1, yy4)
ax2[4].bar(N_1, yy5)

conf = np.concatenate((yy, yy2, yy3, yy4, yy5), axis=0)
conf = np.reshape(conf, (5, 5))
figura, a = plt.subplots()
df_cm = pd.DataFrame(conf, range(5), range(5))
a.set_title('Matriz de Confusão')
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 25})

print("time elapsed: {:.2f}s".format(time.time() - start_time))
plt.show()
