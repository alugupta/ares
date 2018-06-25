import numpy as np
import random
from keras import backend as K

from dl_models.transform import ModelTransform

class Quantize(ModelTransform):
  def __init__(self, layer_mask=None, q=(3,4)):
    super(Quantize,self).__init__(layer_mask)
    self.q            = q

  def __call__(self, model):
    def quantize_wrapper(w):
      def _quantize(q, v):
        (qi,qf) = q
        (imin, imax) = (-np.exp2(qi-1), np.exp2(qi-1)-1)
        fdiv = np.exp2(-qf)
        return np.clip(np.round(v/fdiv)*fdiv, imin, imax)

      ########################################
      shape       = w.shape
      w_flattened = w.flatten()
      w = _quantize(self.q, w_flattened).reshape(shape)

      return w
      ########################################

    ##########################################################################
    self.transform_weights(model,quantize_wrapper)
