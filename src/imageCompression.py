import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# no need for labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
#learning_rate =  3e-5
learning_rate =  1e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()
# initialize layers here
def initialize_weights_m(in_size,out_size,params,name=''):
    W, b = None, None
    low = -np.sqrt(6/(in_size+out_size))
    high = np.sqrt(6/(in_size+out_size)) 
    W = np.random.uniform(low,high,(in_size,out_size))
    b = np.zeros(out_size)
    mw = 0
    Mb = 0
    params['W' + name] = W
    params['b' + name] = b
    params['Mw' + name] = mw
    params['Mb' + name] = Mb



initialize_weights_m(1024,32,params,'layer1')
initialize_weights_m(32,32,params,'hidden')
initialize_weights_m(32,32,params,'hidden2')
initialize_weights_m(32,1024,params,'output')

#new compute loss function
def autoEncoderLoss(inputImage,outputImage):
    return np.square(inputImage - outputImage).sum()

totalLossList = []

for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        h1 = forward(xb,params,'layer1',relu)
        h2 = forward(h1,params,'hidden',relu)
        h3 = forward(h2,params,'hidden2',relu)
        out = forward(h3,params,'output',sigmoid)
        
        loss = autoEncoderLoss(xb,out)
        total_loss += loss
        
        delta1 = 2*(out - xb)
        delta2 = backwards(delta1,params,'output',sigmoid_deriv)
        delta3 = backwards(delta2,params,'hidden2',relu_deriv)
        delta4 = backwards(delta3,params,'hidden',relu_deriv)
        backwards(delta4,params,'layer1',relu_deriv)
        #apply momentum
        params['Mwlayer1'] = 0.9 * params['Mwlayer1'] - learning_rate * params['grad_Wlayer1']
        params['Wlayer1'] += params['Mwlayer1']
        
        params['Mblayer1'] = 0.9 * params['Mblayer1'] - learning_rate * params['grad_blayer1']
        params['blayer1'] += params['Mblayer1']
        
        params['Mwhidden'] = 0.9 * params['Mwhidden'] - learning_rate * params['grad_Whidden']
        params['Whidden'] += params['Mwhidden']
        
        params['Mbhidden'] = 0.9 * params['Mbhidden'] - learning_rate * params['grad_bhidden']
        params['Whidden'] += params['Mbhidden']
        
        params['Mwhidden2'] = 0.9 * params['Mwhidden2'] - learning_rate * params['grad_Whidden2']
        params['Whidden2'] += params['Mwhidden2']
        
        params['Mbhidden2'] = 0.9 * params['Mbhidden2'] - learning_rate * params['grad_bhidden2']
        params['Whidden2'] += params['Mbhidden2']
        
        params['Mwoutput'] = 0.9 * params['Mwoutput'] - learning_rate * params['grad_Woutput']
        params['Woutput'] += params['Mwoutput']
        
        params['Mboutput'] = 0.9 * params['Mboutput'] - learning_rate * params['grad_boutput']
        params['boutput'] += params['Mboutput']
    
    totalLossList.append(total_loss)      
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
#    if itr % lr_rate == lr_rate-1:
#        learning_rate *= 0.9
        
# visualize some results
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(len(totalLossList)),totalLossList,'r')
plt.xlabel('Epoch')
plt.ylabel('Total loss')
plt.title('Total loss v/s epoch')

h1 = forward(xb,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)
for i in range(5):
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(xb[i].reshape(32,32).T,cmap=plt.cm.gray)
    plt.subplot(2,1,2)
    plt.imshow(out[i].reshape(32,32).T,cmap=plt.cm.gray)
    plt.show()

#validate data
valid_y = valid_data['valid_labels']
batches = get_random_batches(valid_x,valid_y,1)

n_class = 0

       
while n_class < 5:
    n_img = 0
    for xb,yb in batches:
        if np.argmax(yb) == int(n_class):
                h1 = forward(xb,params,'layer1',relu)
                h2 = forward(h1,params,'hidden',relu)
                h3 = forward(h2,params,'hidden2',relu)
                out = forward(h3,params,'output',sigmoid)
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

from skimage.measure import compare_psnr as psnr
# evaluate PSNR
psnrTotal = 0
for xb, _ in batches:
    h1 = forward(xb,params,'layer1',relu)
    h2 = forward(h1,params,'hidden',relu)
    h3 = forward(h2,params,'hidden2',relu)
    out = forward(h3,params,'output',sigmoid)
    psnrTotal += psnr(xb,out)
PSNR = psnrTotal/len(valid_x)
print('The average Peak Signal-to-Noise-Ratio is %.2f'%PSNR)
    
    

