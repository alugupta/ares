import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from dl_models.models.tidigits.tidigits_utils import *
import operator as op

from dl_models.models.base import *
import h5py
import sys
import numpy as np

class tidigitsGRUPT(nn.Module):
    def __init__(self):
        self.GRU_feat = 200
        self.seq_size = 254
        self.features  = 39
        self.nb_classes = 10

        super(tidigitsGRUPT, self).__init__()
        self.GRU = nn.GRU(self.features, self.GRU_feat)
        self.fc1 = nn.Linear(self.seq_size * self.GRU_feat, 200)
        self.fc2 = nn.Linear(200, self.nb_classes)

    def forward(self, x):
        batch_size = x.size(0)  
        self.GRU.flatten_parameters()
        x = x.permute([1,0,2])      #Move to seq-batch-samp
        hidden = Variable(torch.rand(1, batch_size, self.GRU_feat, device = x.device))
        output = []
        for i in range(self.seq_size):   #Loop over TIME dimension to allow hidden layer injects
            out, hidden = self.GRU(x[i].unsqueeze(0), hidden)
            output.append(hidden.squeeze(0))

        out = torch.stack(output)
        out = out.permute([1,0,2])  #Return to batch-seq-samp
        out = out.contiguous().view(batch_size,-1)
        out = F.relu(self.fc1(out))
        return F.log_softmax(self.fc2(out), -1)
        

class tidigitsGRU(ModelBase):
  def __init__(self):
    super(tidigitsGRU,self).__init__('tidigits','gru')

    self.param_layer_ids       = ['Dense', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001', '0.001','0.001', '0.001','0.001', '0.001']

    self.lr = .01
    self.l2 = 1e-6

  def build_model(self,):
    model = tidigitsGRUPT()
    self.set_model(model, self.param_layer_ids, self.default_prune_factors)

  def load_dataset(self, ):
    x_train, y_train, x_test, y_test = load_dataset()
    trainset = tidigits(x_train, y_train)
    testset = tidigits(x_test, y_test)
    self.set_data(trainset, testset, testset)

  def compile_model(self, loss='categorical_crossentropy', optimizer='adam', metrics=None):
    super().compile_model(loss=loss, optimizer=optimizer, metrics=metrics)


