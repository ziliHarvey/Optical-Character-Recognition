#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:03:16 2018

@author: zili
"""

import torch
import torchvision
import torchvision.transforms as transforms

import csv
import ipdb
import sys
import numpy as np
import base64
import os
import random
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def getDataLoaders():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True)

    testset = torchvision.datasets.MNIST(root='../data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False)

    return trainloader, testloader

def vizImages(trainloader):
    fig, ax=plt.subplots(1,10)
    #turn off axis for visualization
    for ax_i in ax:
        ax_i.set_xticks([])
        ax_i.set_yticks([])
    class_n = 0
    for (images,labels) in trainloader:
            labels_list = labels.numpy()
            for i in range(len(labels_list)):
                if labels_list[i] == class_n:
                    ax[class_n].imshow(images[i,0].numpy(),cmap='gray')
                    class_n += 1
            if class_n == 10:
                break         


def getClassAcc(testloader, net, batch_size=1):
    total = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    count = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    
    for step,(x,y) in enumerate(testloader):
        b_x = torch.autograd.Variable(x)      
        b_y = torch.autograd.Variable(y)
        output, _, _ = net(b_x)
        pre_y = torch.max(output,1)[1].data.numpy().squeeze()
        act_y = y.numpy().squeeze()
        for i in range(len(pre_y)):
            total[act_y[i]] += 1
            if pre_y[i] == act_y[i]:
                count[pre_y[i]] += 1
    for i in range(10):
        print('The prediction accuracy of label %d is %.2lf%%'%(i,count[i]/total[i]*100))
            
    




def plotLossAcc(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure()
    plt.plot(range(len(train_losses)),train_losses)
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.title('train_loss v/s epoch')
    plt.show()
    
    plt.figure()
    plt.plot(range(len(test_losses)),test_losses)
    plt.xlabel('epoch')
    plt.ylabel('test loss')
    plt.title('test_loss v/s epoch')
    plt.show()
    
    plt.figure()
    plt.plot(range(len(train_accuracies)),train_accuracies)
    plt.xlabel('epoch')
    plt.ylabel('train accuracy')
    plt.title('train_accuracy v/s epoch')
    plt.show()
    
    plt.figure()
    plt.plot(range(len(test_accuracies)),test_accuracies)
    plt.xlabel('epoch')
    plt.ylabel('test accuracy')
    plt.title('test_accuracy v/s epoch')
    plt.show()
    



def vizConvWeights(net):
    w1 = net.conv1[0].weight.data
    n_w1 = len(w1)
    w2 = net.conv2[0].weight.data
    n_w2 = len(w2)
    fig1, ax1=plt.subplots(1,n_w1)
    for ax_i in ax1:
        ax_i.set_xticks([])
        ax_i.set_yticks([])
    for i in range(n_w1):
        ax1[i].imshow(w1[i,0])
        
    fig2, ax2=plt.subplots(1,n_w2)
    for ax_i in ax2:
        ax_i.set_xticks([])
        ax_i.set_yticks([])
    for i in range(n_w2):
        ax2[i].imshow(w2[i,0])
    
    


def vizFeatureMaps(testloader, net):
    for step,(x,y) in enumerate(testloader):
        b_x = torch.autograd.Variable(x)      
        b_y = torch.autograd.Variable(y)
        output, f1, f2 = net(b_x)
        
        n_f1 = f1.data.shape[1]
        fig1, ax1=plt.subplots(1,n_f1)
        for ax_i in ax1:
            ax_i.set_xticks([])
            ax_i.set_yticks([])
        for i in range(n_f1):
            ax1[i].imshow(f1.data[0,i])
            
        n_f2 = f2.shape[1]
        fig2, ax2=plt.subplots(1,n_f2)
        for ax_i in ax2:
            ax_i.set_xticks([])
            ax_i.set_yticks([])
        for i in range(n_f2):
            ax2[i].imshow(f2.data[0,i])  
        break

import torch.nn as nn
import torch.nn.functional as F
import ipdb

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
            nn.Linear(320,50),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(50,10)
        

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f2_flat = f2.view(f2.size(0), -1)
        f3 = self.fc1(f2_flat)
        output = self.fc2(f3)
        return output,f1,f2
    
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

def train(trainloader, optimizer, criterion, epoch, net):
    loss_sum_train = 0
    acc_sum_train = 0
    for step, (x_train, y_train) in enumerate(trainloader):
        b_x_train = torch.autograd.Variable(x_train)      
        b_y_train = torch.autograd.Variable(y_train)
        output_train, _, _ = net(b_x_train)
        loss_train = criterion(output_train,b_y_train)
        acc_sum_train += sum(torch.max(output_train, 1)[1].data.numpy() == y_train.numpy())
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        loss_sum_train += loss_train.data.numpy()
    train_loss = loss_sum_train/len(trainloader.dataset)
    train_acc = acc_sum_train/len(trainloader.dataset)
    print('Epoch: ', epoch, '| train loss: %.4f' % train_loss)
    return train_loss, train_acc   


 

def test(testloader, criterion, net):
    loss_sum =0
    acc_sum = 0
    for step,(x,y) in enumerate(testloader):
        b_x = torch.autograd.Variable(x)      
        b_y = torch.autograd.Variable(y)
        output, _, _ = net(b_x)
        loss = criterion(output,b_y)
        loss_sum += loss.data.numpy()
        pre_y = torch.max(output,1)[1].data.numpy().squeeze()
        act_y = y.numpy().squeeze()
        acc_sum += sum(pre_y == act_y)
    acc = acc_sum/len(testloader.dataset)    
    test_loss = loss_sum/len(testloader.dataset)
    return test_loss, acc



def main():    
    start_epoch = 0
    max_epoch = 10
    learning_rate = 1e-3
    
    trainloader, testloader = getDataLoaders()
    vizImages(trainloader)
    net = Net1()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),learning_rate)
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    #train the network
    for i in range(start_epoch,max_epoch):
        train_loss, train_acc = train(trainloader,optimizer,criterion,i,net)
        test_loss, acc = test(testloader,criterion,net)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(acc)
        
    #final test accuracy
    _,acc_f = test(testloader,criterion,net)
    print('the final accuracy is %.2lf%%'%(100*acc_f))
    
    plotLossAcc(train_losses,test_losses,train_accuracies,test_accuracies)
    vizConvWeights(net)
    vizFeatureMaps(testloader,net)
    getClassAcc(testloader,net)


if __name__=='__main__':

    main()