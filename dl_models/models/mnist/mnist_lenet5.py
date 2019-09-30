import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import operator as op

from dl_models.models.base import *
from functools import reduce

class mnistLenet5PT(nn.Module):
    def __init__(self):
        super(mnistLenet5PT, self).__init__()
        self.nb_classes = 10
        self.nb_filters = 32
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.nb_filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.nb_filters, out_channels=self.nb_filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=.25),
            nn.MaxPool2d(kernel_size=2))

        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, self.nb_classes)
        self.drop = nn.Dropout(p=.5)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,4608)
        x = self.drop(F.relu(self.fc1(x)))
        return F.softmax(self.fc2(x), 1)


class mnistLenet5(ModelBase):
  def __init__(self):
    super(mnistLenet5,self).__init__('mnist','lenet5')

    self.layer_ids             = ['Conv', 'Bias', 'Conv', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001','0.001','0.001','0.001','0.001', '0.001','0.001', '0.001']

    self.l2 = 0.00001  

    
  def load_dataset(self, ):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
    testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)
    self.set_data(trainset, testset, testset)

  def build_model(self,):
    module = mnistLenet5PT()
    self.set_model(module, self.layer_ids, self.default_prune_factors)


