from dl_models.transform import ModelTransform

class Retraining(ModelTransform):
  def __init__(self, layer_mask=None, verbose=False):
    super(Retraining,self).__init__(layer_mask)
    self.verbose = verbose

  @staticmethod
  def set_untrainable(layer):
    layer.trainable = False
  @staticmethod
  def set_trainable(layer):
    layer.trainable = True

  def config(self, model):
    self.transform_layers(model, Retraining.set_untrainable, None)
    self.transform_layers(model, Retraining.set_trainable)
    model.compile_model()

  def reset(self, model):
    self.transform_layers(model, Retraining.set_trainable, None)
    model.compile_model()

  def __call__(self, model, config=True):
    if config:
      self.config(model)
    # Now actually retrain the model
    model.fit_model(v=self.verbose)
    if config:
      self.reset(model)
    return True
