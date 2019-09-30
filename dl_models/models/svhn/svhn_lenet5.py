
import operator as op

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from dl_models.models.base import *
import sys

class svhnLenet5PT(nn.Module):
    def __init__(self):
        super(svhnLenet5PT, self).__init__()
        self.nb_classes  = 10
        self.in_channels = 3
        self.nb_filters  = 32
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.nb_filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=.25),
            nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Sequential(
            nn.Linear(6272, 128),
            nn.ReLU(),
            nn.Dropout(p=.5),
            nn.Linear(128, self.nb_classes))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,6272)
        return F.log_softmax(self.fc(x), dim = -1)

class svhnLenet5(ModelBase):
  def __init__(self):
    super(svhnLenet5,self).__init__('svhn','lenet5')
    self.layer_ids             = []
    self.default_prune_factors = []

  def load_dataset(self, ):
    transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    trainset = torchvision.datasets.SVHN(root='./data',split='train',download=True,transform=transform)
    testset = torchvision.datasets.SVHN(root='./data',split='test',download=True,transform=transform)
    self.set_data(trainset, testset, testset)

  def build_model(self,):
    module = svhnLenet5PT()
    self.set_model(module, self.layer_ids, self.default_prune_factors)

