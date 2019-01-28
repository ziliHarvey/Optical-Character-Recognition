#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 10:40:50 2018

@author: zili
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import scipy.io

#load data
train_data = scipy.io.loadmat('../data/nist36_train.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')
#valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']
#valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

#build a fully-connected network
class SimpleNeuralNet(nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(SimpleNeuralNet, self).__init__()
        self.hidden = nn.Linear(input_nodes,hidden_nodes)   # hidden layer
        self.out = nn.Linear(hidden_nodes,output_nodes)   # output layer


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x
    
#set up parameters
batch_size = 100
input_dimension = 1024
hidden_dimension = 500
output_dimension = 36
learning_rate = 0.001
num_epochs = 100

train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).long()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).long()
print(test_y.shape)
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
print(train_dataset)
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset
)
print(train_dataloader)

neural_net = SimpleNeuralNet(input_dimension,hidden_dimension,output_dimension)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(neural_net.parameters(),learning_rate)
lossList = []
accList = []
for epoch in range(num_epochs):
    loss_sum_train = 0
    acc_sum = 0
    for iteration, (images, labels) in enumerate(train_dataloader):
        x_batch = torch.autograd.Variable(images.view(-1,32,32))
        y_batch = torch.autograd.Variable(labels)
#        print(y_batch.shape)
        output = neural_net(x_batch)
        acc_sum += sum(torch.max(output, 1)[1].data.numpy() == torch.max(y_batch, 1)[1].data.numpy())
#        print(output.shape)
        loss = criterion(output,torch.max(labels, 1)[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum_train += loss.data.numpy()
    lossList.append(loss_sum_train)
    accList.append(acc_sum/len(train_dataloader.dataset))

plt.figure()
plt.plot(range(num_epochs),lossList)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss v/s Epoch')

plt.figure()
plt.plot(range(num_epochs),accList)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train Accuracy v/s Epoch')
        
count = 0
num = len(test_dataloader)
for (images, labels) in test_dataloader:
    images = torch.autograd.Variable(images.view(-1,32,32))
    pred = neural_net(images)
    predy = torch.max(pred, 1)[1].data.numpy().squeeze()
    testy = torch.max(labels, 1)[1].numpy()
    if predy == testy:
        count += 1
accuracy = count/num*100
print('The model accuracy on test set is: %.2f%%' % accuracy)

















