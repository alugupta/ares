import operator as op

from dl_models.models.base import *
import sys
from functools import reduce

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class cifarAlexPT(nn.Module):
    def __init__(self):
        super(cifarAlexPT, self).__init__()
        self.nb_classes  = 10

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.fc = nn.Sequential(
            nn.Dropout(p=.2),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Dropout(p=.2),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.nb_classes))
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 1024)
        return self.fc(x)

class cifar10alexnet(ModelBase):
  def __init__(self):
    super(cifar10alexnet,self).__init__('cifar','alexnet')
    self.layer_ids             = []
    self.default_prune_factors = []

    self.l2 = .1
    self.lr = 1e-4

  def load_dataset(self, ):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    self.set_data(trainset, testset, testset)
  
  def compile_model(self, loss='categorical_crossentropy', optimizer='adadelta', metrics=None):
    super().compile_model(loss, optimizer, metrics)

  def build_model(self,faults=[]):
    module = cifarAlexPT()
    self.set_model(module, self.layer_ids, self.default_prune_factors)
    


