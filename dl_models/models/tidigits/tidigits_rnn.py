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

class tidigitsRNNPT(nn.Module):
    def __init__(self):
        super(tidigitsRNNPT, self).__init__()
        self.RNN_feat = 1024
        self.seq_size = 254
        self.features  = 39
        self.nb_classes = 10

        self.RNN = nn.RNN(self.features, self.RNN_feat)
        self.fc1 = nn.Linear(self.seq_size*self.RNN_feat, 100)
        self.fc2 = nn.Linear(100, self.nb_classes)

    def forward(self, x):
        batch_size = x.size(0)

        hidden = Variable(torch.rand(1, batch_size, self.RNN_feat, device = x.device))
        self.RNN.flatten_parameters()
        output = []
        x = x.permute([1,0,2])      #Move to seq-batch-samp
        for i in range(self.seq_size):   #Loop over TIME dimension to allow hidden layer injects
            out, hidden = self.RNN(x[i].unsqueeze(0), hidden)
            output.append(out.squeeze(0))
        out = torch.stack(output)
        out = out.permute([1,0,2])
        out = out.contiguous().view(batch_size, -1)  #Return to batch-seq-samp

        out = F.relu(self.fc1(out))
        return F.log_softmax(self.fc2(out), -1)

class tidigitsRNN(ModelBase):
  def __init__(self):
    super(tidigitsRNN,self).__init__('tidigits','rnn')

    self.param_layer_ids       = ['Dense', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001', '0.001','0.001', '0.001','0.001', '0.001']

    self.lr = .0001

  def build_model(self,):
    model = tidigitsRNNPT()
    self.set_model( model, self.param_layer_ids, self.default_prune_factors )

  def load_dataset(self, ):
    x_train, y_train, x_test, y_test = load_dataset()
    trainset = tidigits(x_train, y_train)
    testset = tidigits(x_test, y_test)
    self.set_data(trainset, testset, testset)


  def compile_model(self, loss='categorical_crossentropy', optimizer='sgd', metrics=None):
    super().compile_model(loss=loss, metrics=metrics)
    self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr, momentum=0.9, nesterov=False)
