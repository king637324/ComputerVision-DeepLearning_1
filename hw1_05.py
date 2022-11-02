from sys import int_info
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from time import sleep

import ui_05 as ui

from PyQt5.QtWidgets import QMainWindow, QApplication
import sys
import os
import cv2
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchsummary import summary


numClasses = 10
batch_size = 256
learning_rate = 0.01
op = 'SGD'
epochs =80

optimizers = {'SGD':torch.optim.SGD, 'Adam':torch.optim.Adam, 'RMSprop':torch.optim.RMSprop}


transform = transforms.Compose([
    transforms.ToTensor(), # 0 1
])

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

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(512 * 1 * 1, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4096, numClasses)
        )

    def forward(self, x):
        x = self.vgg16(x)
        return x

class torchDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


model = VGG16(numClasses=numClasses)

use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

optimizer = optimizers[op](model.parameters(), lr=learning_rate, momentum=0.9)

class Main(QMainWindow, ui.Ui_mainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.ButtonQ5_1.clicked.connect(self.Q5_1)
        self.ButtonQ5_2.clicked.connect(self.Q5_2)
        self.ButtonQ5_3.clicked.connect(self.Q5_3)
        self.ButtonQ5_4.clicked.connect(self.Q5_4)
        self.ButtonQ5_5.clicked.connect(self.Q5_5)

        self.labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.testDataset = None
        self.vgg16_model = None

        self.read_cifar10_vgg16()
    
    def read_cifar10_vgg16(self):
        data_path = r'cifar10'
        train_data = []
        train_label = []

        for i in range(1, 6):
            with open(data_path + r'\data_batch_' + str(i), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                train_data.append(data[b'data'])
                train_label.append(data[b'labels'])

        with open(data_path + r'\test_batch', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            test_data = data[b'data']
            test_label = data[b'labels']

        train_data = np.array(train_data).reshape((-1, 3, 32, 32))
        test_data = np.array(test_data).reshape((-1, 3, 32, 32))

        self.train_data = train_data.transpose((0, 2, 3, 1))
        self.train_label = np.array(train_label).ravel()

        self.test_data = test_data.transpose((0, 2, 3, 1))
        self.test_label = np.array(test_label)
        self.testDataset = torchDataset(self.test_data, self.test_label, transform=transform)
        self.vgg16_model = VGG16(numClasses=numClasses)
        self.vgg16_model = torch.load('vgg16.pkl', map_location=torch.device('cpu'))
        self.vgg16_model.eval()

    def Q5_1(self):
        print("-------------Q5_1-------------")

        if len(self.train_data) == 0:
            self.read_cifar10_vgg16()
        
        idx = random.sample([i for i in range(len(self.train_data))], 9)

        # figsize 設定視窗大小
        plt.figure(figsize=(6, 6))
        for i in range(9):
            # nrows：行數、ncols：列數、sharex：和誰共享x軸
            plt.subplot(3, 3, i+1)

            plt.xticks([])
            plt.yticks([])
            plt.title(self.labels[self.train_label[idx[i]]], fontdict={'color': 'black'})
            plt.imshow(self.train_data[idx[i]])

        plt.tight_layout()
        plt.show()

        print("-------------Q5_1 Finsh-------------\n")
    
    def Q5_2(self):
        print("-------------Q5_2-------------")

        print('hyperparameters:')
        print('Batch Size: ', batch_size)
        print()
        print('optimizer: ', optimizer)

        print("-------------Q5_2 Finsh-------------\n")
    
    def Q5_3(self):
        print("-------------Q5_3-------------")

        summary(model, input_size=(3, 32, 32))

        print("-------------Q5_3 Finsh-------------\n")
    
    def Q5_4(self):
        print("-------------Q5_4-------------")
        cv2.imshow('Training loss', cv2.imread('train_loss.png'))

        cv2.imshow('Training accuracy', cv2.imread('train_acc.png'))

        print("-------------Q5_4 Finsh-------------\n")
    
    def Q5_5(self):
        print("-------------Q5_5-------------")

        if len(self.test_data) == 0:
            self.read_cifar10_vgg16()
        
        test_index = self.spinBox.value()

        data, label = self.testDataset[int(test_index)]

        img = data.numpy().transpose(1, 2, 0)
        data = torch.unsqueeze(data, 0)

        out = self.vgg16_model(data)
        softmax = torch.nn.Softmax(dim=1)
        softmax = softmax(out).tolist()[0]
        pred = torch.max(out.data, 1)[1]
        ans = self.labels[label.item()]
        pred = self.labels[pred.item()]

        plt.figure(figsize=(10, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('target=' + ans + ', predict=' + pred, fontdict={'color': 'black'})

        plt.subplot(1, 2, 2)
        plt.bar([i for i in range(10)], softmax, align='center')
        plt.xticks(range(10), self.labels, fontsize=10, rotation=0)
        plt.yticks((0.2, 0.4, 0.6, 0.8, 1))
        plt.tight_layout()

        plt.savefig('predict.png')
        plt.close()
        cv2.imshow('Test Image index=' + str(test_index), cv2.imread('predict.png'))
        os.remove('predict.png')


        print("-------------Q5_5 Finsh-------------\n")

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())