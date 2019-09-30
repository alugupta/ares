import operator as op

from dl_models.models.base import *
import sys
from functools import reduce

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
#ADAPTED FROM https://github.com/chengyangfu/pytorch-vgg-cifar10
class cifarVGGPT(nn.Module):
    def __init__(self):
        super(cifarPT, self).__init__()
        self.features = self._make_layers(cfg)
        self.class1 = nn.Linear(2048, 512)
        self.class2 = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.class2(F.relu(self.class1(out)))
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class cifar10VGG(ModelBase):
  def __init__(self):
    super(cifar10VGG,self).__init__('cifar','CiFar10VGG')
    self.layer_ids             = []
    self.default_prune_factors = []

    self.l2 = .1
    self.lr = 5e-4
    
  def fit_model(self, batch_size=128, v=0): #Override for LR decay  #TODO: CHECKW
    self.lr = .1
    self.model.train()
    trainloader = torch.utils.data.DataLoader(self.traindata, batch_size=batch_size, shuffle=True, num_workers=4)
    self.model.to(self.device)
    for e in range(self.num_epochs):
        print("Training epoch " + str(e))
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        if e % 60 == 0 and e > 0:
            self.lr /= 10
            print(self.test_model())
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr, weight_decay = self.l2, momentum=0.9, nesterov=True)

  def load_dataset(self, ):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)
    self.set_data(trainset, testset, testset)

  def compile_model(self, loss='categorical_crossentropy', optimizer='sgd', metrics=None):
    super().compile_model(loss, optimizer, metrics)

  def build_model(self,faults=[]):
    module = cifarVGGPT()
    self.set_model(module, self.layer_ids, self.default_prune_factors)

    


