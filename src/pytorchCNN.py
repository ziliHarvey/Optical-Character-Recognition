#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:09:33 2018

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

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,10,5),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10,20,5),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(500,200),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(200,36)
    
        

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f2_flat = f2.view(f2.size(0), -1)
        f3 = self.fc1(f2_flat)
        output = self.fc2(f3)
        return output    
#set up parameters
batch_size = 50
learning_rate = 0.001
num_epochs = 50

train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).long()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).long()
print(test_y.shape)
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset
)
neural_net = Net1()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(neural_net.parameters(),learning_rate)
lossList = []
accList = []
for epoch in range(num_epochs):
    loss_sum_train = 0
    acc_sum = 0
    for iteration, (images, labels) in enumerate(train_dataloader):
        x_batch = torch.autograd.Variable(images)
        x_batch = x_batch.reshape(-1,1,32,32)
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
    print('Epoch: ', epoch, '| train loss: %.4f' % loss_sum_train)

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
    images = images.reshape(-1,1,32,32)
    images = torch.autograd.Variable(images)
    pred = neural_net(images)
    predy = torch.max(pred, 1)[1].data.numpy().squeeze()
    testy = torch.max(labels, 1)[1].numpy()
    if predy == testy:
        count += 1
accuracy = count/num*100
print('The model accuracy on test set is: %.2f%%' % accuracy)

