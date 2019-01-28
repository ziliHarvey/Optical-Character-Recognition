import numpy as np
from util import *

# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None
    low = -np.sqrt(6/(in_size+out_size))
    high = np.sqrt(6/(in_size+out_size)) 
    W = np.random.uniform(low,high,(in_size,out_size))
    b = np.zeros(out_size)
    params['W' + name] = W
    params['b' + name] = b

# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None
    res = 1/(1+np.exp(-x))
    return res

def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]
    
    # your code here
    pre_act = X@W + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act
 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    c = -np.array([np.max(x,1)]).reshape(x.shape[0],1)
    den = np.array([np.sum(np.exp(x+c),1)]).reshape(x.shape[0],1)
    res = np.exp(x+c)/den            
    return res

# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    #calculate loss
    loss = -(y*np.log(probs)).sum()
    #calculate accuracy
    n_correct = 0
    for i in range(y.shape[0]):
        if np.argmax(probs[i,:]) == np.argmax(y[i,:]):
            n_correct += 1
    acc = n_correct/y.shape[0]
                
    return loss, acc 

# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    grad_act = delta*activation_deriv(post_act)
    grad_b = np.sum(grad_act,axis = 0)
    grad_W = X.T@grad_act
    grad_X = grad_act@W.T 
    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    #combine x, y together
    data = np.hstack((x,y))
    #shuffle instances
    np.random.shuffle(data)
    num = np.size(data,0)
    dim_x = x.shape[1]
    for i in range(int(num/batch_size)):
        batch_x = data[int(i*batch_size):int((i+1)*batch_size),:dim_x]
        batch_y = data[int(i*batch_size):int((i+1)*batch_size),dim_x:]
        batches.append((batch_x,batch_y))
    return batches
