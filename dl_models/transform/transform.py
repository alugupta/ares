# Generic model transformation base

from abc import abstractmethod
import numpy as np
import torch


class ModelTransform(object):
  def __init__(self, layer_mask=None):
    self.layer_mask = layer_mask

  @abstractmethod
  def __call__(self, model):
    return model

  @staticmethod
  def _generalize_layer_mask(model,layer_mask=None):
    # Several ways to specify which layers to transform:
    #   None: transform every layer
    #   <int>: transform only layers starting with the nth
    #   <list>: transform specific layers (must match length of model layer list)
    chosen_layers = 1+np.zeros(len(model.get_layers()),dtype='bool')
    if layer_mask is None:
      return chosen_layers
    else:
      try:
        if len(layer_mask):
          is_list = True
      except TypeError:
        is_list = False
      if is_list:
        chosen_layers = np.array(layer_mask)
      else:
        chosen_layers[0:layer_mask] = 0
    assert len(model.get_layers())==len(chosen_layers), 'Layer mask must match number of layers. ('+str(len(model.get_layers()))+' vs. '+str(len(chosen_layers))+')'
    return chosen_layers
    
  #Take a user-provided layer mask and convert to one that includes biases
  def expand_mask(self, layer_mask, skip_biases, model): 
    layers = model.get_all_layers()
    new_mask = np.zeros(len(layers),dtype='bool')
    j = 0
    for i in range(len(layers)):
        if 'bias' not in layers[i][0]:
            new_mask[i] = layer_mask[j]
            j += 1
        elif not skip_biases and j > 0:   #If skipping, ignore. If not, copy the status of previous layer
            new_mask[i] = layer_mask[j - 1]

    return new_mask
        

  def get_masked_layers(self, model, layer_mask=-1):
    if layer_mask is -1: # allow overriding the mask with a non-sentinel
      layer_mask = self.layer_mask
    mask = self._generalize_layer_mask(model, layer_mask)
    mask = self.expand_mask(mask, skip_biases, model)

    layers = model.get_all_layers()
    masked_layers = [l for l,m in zip(layers,mask) if m]
    return masked_layers

  def transform_layers(self, model, function, skip_biases, layer_mask=-1):
    if layer_mask is -1: # allow overriding the mask with a non-sentinel
      layer_mask = self.layer_mask
    mask = self._generalize_layer_mask(model, layer_mask)
    mask = self.expand_mask(mask, skip_biases, model)

    layers = model.get_all_layers()
    for l,m in zip(layers, mask):
      if m:
        v = function(l) # Must mutate the layer in-place
        assert v is None, 'Layer transform function must do their work in-place.'
    return model

  def transform_weights(self, model, function, layer_mask=-1, skip_biases=False):
    def sub_transform(layer):
      new_weights = function(layer[1].data)
      model.update_layer(layer, new_weights) 
      return None # Mutation is in-place
    return self.transform_layers(model, sub_transform, skip_biases, layer_mask=layer_mask)

