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
import time

from dl_models.models.imagenet.imagenet_utils import *

import random
import sys
import numpy

class imagenetBase(ModelBase):
  def __init__(self, dataset='imagenet', model_name='base_model'):
    super(imagenetBase,self).__init__(dataset,model_name)

    self.num_train_blocks = 5004
    self.num_val_blocks   = 179

    self.param_layer_ids = []
    self.default_prune_factors = []

    #Directory to store preprocessed files. This will need to be changed to ones own download of imagenet. We assume the set has been normalized to [0,1] with var=1
    self.train_dir = '/data/imagenet/train/hickle/'
    self.val_dir = '/data/imagenet/val/pytorch/'
    preprocessing_dir = '/data/imagenet/preprocessed/'
    labels_dir          = preprocessing_dir + "labels/"
    self.val_labels_filepath   = labels_dir + "val_labels.npy"
    self.train_labels_filepath = labels_dir + "train_labels.npy"

  def fit_model(self, batch_size=256, v=0, keep_best=False):
    self.model.to(self.device)
    if self.pretrained:
        return
    best_acc = 1
    
    self.model.train()
    for e in range(self.num_epochs):
        print("Training epoch " + str(e))
        #Generate random batch order
        shuffled_batch_ids = list(range(self.num_train_blocks))
        random.shuffle(shuffled_batch_ids)
        for batch in shuffled_batch_ids:
            X, y = self.traindata.__getitem__(batch)    #Get batch from train or test dataset (stored in 256 chunks)
            inputs, labels = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)        
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        if keep_best:
            best_acc = self.save_best(best_acc)

  def check_data(self, inputs, labels, losses):
    if 'accuracy' in self.metrics:
        output = self.model(inputs).data  
        losses.append(self.accuracy(output, labels.data))

  def check_model(self, dataset):
    self.model.eval()
    losses = []
    
    with torch.no_grad():
        count = 0
        for i in range(self.num_val_blocks):
            diff = time.time()
            X, y = self.testdata.__getitem__(i)          
            inputs, labels = X.to(self.device), y.to(self.device)
            count += list(labels.data.size())[0]
            self.check_data(inputs, labels, losses) 
            print("Testing batch: ", str(i), "TIME: ",time.time()-diff, "ERR: ", 1. - np.sum(losses) / count)
            
        return 1. - np.sum(losses) / count

  def load_dataset(self, ):
    self.train_labels, self.val_labels = load_imagenet_labels(self.train_labels_filepath,self.val_labels_filepath)
    trainset = imageNet(self.train_dir, self.train_labels)
    testset = imageNet(self.val_dir, self.val_labels)

    self.set_data(trainset, testset, testset)

