import torch
import numpy as np
from dl_models.transform import ModelTransform

class SummarizeSparsity(ModelTransform):
  modes = ['fraction','count','both']
  def __init__(self, layers=None, mode='fraction'):
    super(SummarizeSparsity,self).__init__(layers)
    assert mode in self.modes, 'Mode must be one of: '+','.join(self.modes)
    self.mode = mode
    self.summary = []
    self.total_weights = 0
    self.total_nonzero = 0

  def __call__(self, model):
    '''Returns the fraction of nonzeros in weight matrices'''
    self.summary = []
    def sparsity(w):
      self.total_weights += w.numel()
      nz = (w != 0).sum().item()
      self.total_nonzero += nz    
      if self.mode=='fraction':
        c = nz/w.numel()
      elif self.mode=='count':
        c = nz
      elif self.mode=='both':
        c = ( nz/w.numel(),nz )
      self.summary.append(c)    

      return w
    return self.transform_weights(model,sparsity)

  def get_summary(self):
    return self.summary

class SummarizeDistribution(ModelTransform):
  def __init__(self, layers=None):
    super(SummarizeDistribution,self).__init__(layers)
    self.summary = []

  def __call__(self, model):
    '''Returns the mean and std dev of weight matrices'''
    self.summary = []
    def distrib(w):
      self.summary.append( (torch.mean(w).item(),torch.std(w).item()) )
      return w
    return self.transform_weights(model,distrib)

  def get_summary(self):
    return self.summary
