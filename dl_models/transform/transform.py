# Generic model transformation base

from abc import abstractmethod
import numpy as np

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

  def get_masked_layers(self, model, layer_mask=-1):
    if layer_mask is -1: # allow overriding the mask with a non-sentinel
      layer_mask = self.layer_mask
    layers = model.get_layers()
    mask = self._generalize_layer_mask(model, layer_mask)
    masked_layers = [l for l,m in zip(layers,mask) if m]
    return masked_layers

  def transform_layers(self, model, function, layer_mask=-1):
    if layer_mask is -1: # allow overriding the mask with a non-sentinel
      layer_mask = self.layer_mask
    layers = model.get_layers()
    mask = self._generalize_layer_mask(model, layer_mask)
    for l,m in zip(layers, mask):
      if m:
        v = function(l) # Must mutate the layer in-place
        assert v is None, 'Layer transform function must do their work in-place.'
    return model

  def get_masked_weights(self, model, layer_mask=-1, skip_biases=True):
    layers = self.get_masked_layers(model, layer_mask=layer_mask)
    if skip_biases:
      weights = [l.get_weights()[0] for l in layers]
    else:
      weights = [w for l in layers for w in l.get_weights()]
    return weights

  def transform_weights(self, model, function, layer_mask=-1, skip_biases=True):
    def sub_transform(layer):
      weights = layer.get_weights()
      new_weights = []
      for ii, w in enumerate(weights):
        # skip biases for now..
        # or not skip_biases:
        if ii == 0:
          new_w = function(w)
        else:
          new_w = w
        new_weights.append(new_w)
      layer.set_weights(new_weights)
      return None # Mutation is in-place
    return self.transform_layers(model, sub_transform, layer_mask=layer_mask)

