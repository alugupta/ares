from dl_models.transform import ModelTransform

class Retraining(ModelTransform):
  def __init__(self, layer_mask=None, verbose=False):
    super(Retraining,self).__init__(layer_mask)
    self.verbose = verbose

  @staticmethod
  def set_untrainable(layer, skip_biases):
    if 'weight' in layer[0] or not skip_biases:    
        layer[1].requires_grad = False
  @staticmethod
  def set_trainable(layer, skip_biases):
    if 'weight' in layer[0] or not skip_biases:    
        layer[1].requires_grad = True

  def config(self, model):
    self.transform_layers(model, Retraining.set_untrainable, False, None)
    self.transform_layers(model, Retraining.set_trainable, False)
    model.compile_model()

  def reset(self, model):
    self.transform_layers(model, Retraining.set_trainable, False, None)
    model.compile_model()

  def __call__(self, model, config=True):
    if config:
      self.config(model)
    # Now actually retrain the model
    model.fit_model(v=self.verbose)
    if config:
      self.reset(model)
    return True
