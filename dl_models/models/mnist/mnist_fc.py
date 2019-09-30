import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np


import operator as op

from dl_models.models.base import *

from functools import reduce

class mnistFCPT(nn.Module, ):
        def __init__(self):
            super(mnistFCPT, self).__init__()
            self.nb_classes = 10
            self.img_rows = 28
            self.img_cols = 28
            
            self.classifier = nn.Sequential(
                nn.Linear(self.img_rows*self.img_cols, 300),
                nn.ReLU(),
                nn.Linear(300, 100),
                nn.ReLU(),
                nn.Linear(100, self.nb_classes))

        def forward(self, x):
            x = x.view(-1, self.img_rows*self.img_cols)
            x = self.classifier(x)
            return F.softmax(x,dim=1)

class mnistFC(ModelBase):

  def __init__(self):
    super(mnistFC, self).__init__('mnist', 'fc')  
    self.param_layer_ids             = ['Dense', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001','0.001','0.001','0.001','0.001', '0.001']
    self.relu_threshold = 0.0  # Default off
    self.l2 = .00001

  def load_dataset(self, ):  
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
    testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)
    self.set_data(trainset, testset, testset)

  def build_model(self,):   
    module = mnistFCPT()
    self.set_model(module, self.param_layer_ids, self.default_prune_factors)



