import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 10
learning_rate = 1e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,train_y.shape[1],params,'output')


# with default settings, you should get loss < 150 and accuracy > 80%
lossTrainList = []
lossValidateList = []
accTrainList = []
accValidateList = []
import copy
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        # loss
        loss, acc = compute_loss_and_acc(yb, probs)
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss
        total_acc += acc
        # backward
        yb_idx = []
        for i in range(np.size(yb,0)):
            for j in range(np.size(yb,1)):
                if yb[i,j] == 1:
                    yb_idx.append(j)
        delta1 = probs
        delta1[np.arange(probs.shape[0]),yb_idx] -= 1
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)
        # apply gradient
        params['Wlayer1'] += -learning_rate*params['grad_Wlayer1']
        params['blayer1'] += -learning_rate*params['grad_blayer1']
        params['Woutput'] += -learning_rate*params['grad_Woutput']
        params['boutput'] += -learning_rate*params['grad_boutput']
    total_acc = total_acc/len(batches)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
    lossTrainList.append(total_loss)
    accTrainList.append(total_acc)
    #run validation
    params_v = copy.deepcopy(params)  
    h1_v = forward(valid_x,params_v,'layer1')
    probs_v = forward(h1_v,params_v,'output',softmax)
    loss_v, valid_acc = compute_loss_and_acc(valid_y, probs_v)
    lossValidateList.append(loss_v)
    accValidateList.append(valid_acc)
# run on validation set and report accuracy! should be above 75%
h1_v = forward(valid_x,params,'layer1')
probs_v = forward(h1_v,params,'output',softmax)
loss_v, valid_acc = compute_loss_and_acc(valid_y, probs_v)
print('Validation accuracy: ',valid_acc)
print('validation loss: ', loss_v/valid_x.shape[0])
#plot acc-epoch
import matplotlib.pyplot as plt
plt.figure()
epochList = [i for i in range(max_iters)]
plt.plot(epochList,accTrainList,label='Train')
plt.plot(epochList,accValidateList,label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs Epoch')
plt.show()
#plot avg_loss-epoch
plt.figure()
lossTrainList = [i/train_x.shape[0] for i in lossTrainList]
lossValidateList = [i/valid_x.shape[0] for i in lossValidateList]
plt.plot(epochList,lossTrainList,label='Train')
plt.plot(epochList,lossValidateList,label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.title('Average Loss vs Epoch')
plt.show()

if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
#visualize the first layer weights
params_ini = {}
initialize_weights(train_x.shape[1],hidden_size,params_ini,'layer1')
initialize_weights(hidden_size,train_y.shape[1],params_ini,'output')

weightsMatrix_ini = params_ini['Wlayer1'].reshape(32,32,-1)
F_ini = plt.figure()
grid_ini = ImageGrid(F_ini,111,(8,8),axes_pad=0.1)
for i in range(64):
    grid_ini[i].imshow(weightsMatrix_ini[:,:,i],cmap=plt.cm.gray)
    
weightsMatrix_fin = params['Wlayer1'].reshape(32,32,-1)
F_fin = plt.figure()
grid_fin = ImageGrid(F_fin,111,(8,8),axes_pad=0.1)
for i in range(64):
    grid_fin[i].imshow(weightsMatrix_fin[:,:,i],cmap=plt.cm.gray)



# plot confusion matrix 
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
test_data = scipy.io.loadmat('../data/nist36_test.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']
h1_test = forward(test_x,params,'layer1')
probs_test = forward(h1_test,params,'output',softmax)
_, acc = compute_loss_and_acc(test_y,probs_test)
print('test accuracy: ',acc)
for i in range(test_y.shape[0]):
    confusion_matrix[np.argmax(probs_test[i,:]),np.argmax(test_y[i,:])] += 1
import string
plt.figure()
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()