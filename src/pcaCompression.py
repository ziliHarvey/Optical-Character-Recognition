import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# no need for labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA
u, sigma, vt = np.linalg.svd(train_x)
v = vt.T
# rebuild a low-rank version
lrank = None
lrank = train_x @ v[:,:dim]
# rebuild it
recon = None
recon = lrank @ (v[:,:dim]).T


for i in range(5):
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(train_x[i].reshape(32,32).T,cmap=plt.cm.gray)
    plt.subplot(2,1,2)
    plt.imshow(recon[i].reshape(32,32).T,cmap=plt.cm.gray)
    plt.show()

# build valid dataset
valid_y = valid_data['valid_labels']
batches = get_random_batches(valid_x,valid_y,1)
n_class = 0

       
while n_class < 5:
    n_img = 0
    for xb,yb in batches:
        if np.argmax(yb) == int(n_class):
                lrank = xb @ v[:,:dim]
                # rebuild it
                out = lrank @ (v[:,:dim]).T

                plt.figure()
                plt.subplot(2,1,1)
                plt.imshow(xb.reshape(32,32).T,cmap=plt.cm.gray)
                plt.subplot(2,1,2)
                plt.imshow(out.reshape(32,32).T,cmap=plt.cm.gray)
                plt.show()
                n_img += 1
                n_class += 0.5
                if n_img == 2:
                    break
                
recon_valid = None
lrank = valid_x @ v[:,:dim]
recon_valid = lrank @ (v[:,:dim]).T
total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())