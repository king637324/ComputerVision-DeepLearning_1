from sys import int_info
from time import sleep

import numpy as np
import glob
import matplotlib.pyplot as plt

import os
import cv2
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchsummary import summary

import warnings

numClasses = 10
batch_size = 256
learning_rate = 0.01
op = 'SGD'
epochs =80

optimizers = {'SGD':torch.optim.SGD, 'Adam':torch.optim.Adam, 'RMSprop':torch.optim.RMSprop}


# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose([transforms.ToTensor()])
trainDataset = torchvision.datasets.CIFAR10(root = './', train = True, download = True, transform = transform)
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = batch_size, shuffle = True, num_workers = 0)
testDataset = torchvision.datasets.CIFAR10(root = './', train = False, download = True, transform = transform)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size = batch_size, shuffle = True, num_workers = 0)


class VGG16(nn.Module):
    def __init__(self, numClasses=10):
        super(VGG16, self).__init__()
        self.vgg16 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 4096), nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, numClasses)
        )
        
    def forward(self, x):
        x = self.vgg16(x)
        return x



model = VGG16(numClasses=numClasses)

use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

optimizer = optimizers[op](model.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()
summary(model, input_size=(3, 32, 32))

from tqdm import tqdm
from IPython.display import clear_output

loss_epochs = []
train_acc_epochs = []
test_acc_epochs = []

for epoch in range(epochs):
    loss_batch = 0.0
    acc_batch = 0.0
    model.train()
    for i, data in tqdm(enumerate(trainLoader, 1)):
        inputs, labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        inputs = Variable(inputs)
        labels = Variable(labels)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        
        
        acc_batch += (predicted == labels).sum().item()
        loss_batch += loss.item()
        
        #print('[%d, %5d] loss: %.3f' % (epoch, i * batch_size, loss))
        #clear_output(wait=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #break
        
        
    loss_batch /= i
    acc_batch /= (i * batch_size)

    
    print("[%2d, %5d] Train Loss: %.3f" % (epoch + 1, i * batch_size, loss_batch))
    print("[%2d, %5d] Train Accuracy: %.3f" % (epoch + 1, i * batch_size, acc_batch))

    loss_epochs.append(loss_batch)
    train_acc_epochs.append(acc_batch)

    model.eval()
    loss_batch = 0.0
    acc_batch = 0.0

    for i, data in enumerate(testLoader, 1):
        inputs, labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs = Variable(inputs, volatile=True)
        labels = Variable(labels, volatile=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        num_correct = (predicted == labels).sum().item()

        loss_batch += loss.item() 
        acc_batch += num_correct

    loss_batch /= i
    acc_batch /= (i * batch_size)
    test_acc_epochs.append(acc_batch)

    print("[%2d, %5d] Test Loss: %.3f" % (epoch + 1, i * batch_size, loss_batch))
    print("[%2d, %5d] Test Accuracy: %.3f" % (epoch + 1, i * batch_size, acc_batch))

print(loss_epochs)
print(train_acc_epochs)
print(test_acc_epochs)

# 用 COLAB GPU
# torch.save(model, './drive/MyDrive/成大/vgg16_10280250.pkl',_use_new_zipfile_serialization=False)
# 用電腦本地CPU
torch.save(model, 'D:/成大/碩一/電腦視覺與深度學習/Homework/Hw1/vgg16_10280250.pkl',_use_new_zipfile_serialization=False)