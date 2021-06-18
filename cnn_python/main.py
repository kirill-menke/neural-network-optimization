from Layers import *
from Layers import ExpandDims
from Optimization import *
import numpy as np
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
import NeuralNetwork
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class MNISTData():
    def __init__(self, batch_size):
        self.mean = 33.40891
        self.std = 78.67774
        self.batch_size = batch_size
        self.curr_idx = 0

        data = pd.read_csv('../data/mnist-train.txt', sep='\t', header=None).values
        self.images = data[:, 1:].reshape(-1, 1, 28, 28)
        
        labels = data[:, 0]
        self.one_hot_encoded = np.zeros((len(data), 10))
        self.one_hot_encoded[np.arange(len(data)), labels] = 1


    def next(self):
        labels = self.one_hot_encoded[self.curr_idx: self.curr_idx + self.batch_size]
        images = (self.images[self.curr_idx: self.curr_idx + self.batch_size] - self.mean) / self.std
        self.curr_idx += self.batch_size
        return images, labels

    

net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(5e-3),
                                          Initializers.He(),
                                          Initializers.He())
net.data_layer = MNISTData(32)
net.loss_layer = Loss.CrossEntropyLoss()

conv1 = Conv.Conv((1, 1), (1, 3, 3), 4)
relu1 = ReLU.ReLU()
pool1 = Pooling.Pooling((2, 2), (2, 2))
conv2 = Conv.Conv((1, 1), (4, 3, 3), 8)
relu2 = ReLU.ReLU()
pool2 = Pooling.Pooling((2, 2), (2, 2))
flatten1 = Flatten.Flatten()
expand_dims1 = ExpandDims.ExpandDims()
conv3 = Conv.Conv((1, 1), (8*7*7, 1, 1), 10)
flatten2 = Flatten.Flatten()
softmax = SoftMax.SoftMax()


net.append_layer(conv1)
net.append_layer(relu1)
net.append_layer(pool1)
net.append_layer(conv2)
net.append_layer(relu2)
net.append_layer(pool2)
net.append_layer(flatten1)
net.append_layer(expand_dims1)
net.append_layer(conv3)
net.append_layer(flatten2)
net.append_layer(softmax)

net.train(5000)