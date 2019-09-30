import numpy as np
import random
import torch

from dl_models.transform import ModelTransform

class Quantize(ModelTransform):
  def __init__(self, layer_mask=None, q=(3,4)):
    super(Quantize,self).__init__(layer_mask)
    self.q = q

  def __call__(self, model):
    def quantize_wrapper(w):
      qi, qf = self.q
      (imin, imax) = (-np.exp2(qi-1), np.exp2(qi-1)-1) 
      fdiv = np.exp2(-qf)
      w = torch.mul(torch.round(torch.div(w, fdiv)), fdiv)
      return torch.clamp(w, min=imin, max=imax)
      ########################################

    ##########################################################################
    self.transform_weights(model,quantize_wrapper)
