import operator as op

from dl_models.models.base import *

import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import scipy
import torch

from dl_models.models.imagenet.imagenet_utils import *
from dl_models.models.imagenet.imagenet_base import *

import random
import sys
import numpy

class imagenetInceptionv3(imagenetBase):
  def __init__(self):
    super(imagenetInceptionv3,self).__init__('imagenet','inceptionv3')

    self.param_layer_ids = []
    self.default_prune_factors = []
    
    self.lr = .0001 
    self.l2 = 1e-6

  def build_model(self, pretrained = True):
    self.pretrained = pretrained
    model = models.inception_v3(pretrained=pretrained)
    self.set_model(model, self.param_layer_ids, self.default_prune_factors)

  # overriding compile_model from base as loss should be mse and
  # optimizer sgd (work slightly better)
  def compile_model(self, loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy','t5']):
    super().compile_model(loss=loss,optimizer=optimizer, metrics=metrics)

