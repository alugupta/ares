import os.path
import numpy as np

from dl_models.models.model_configs import *

from dl_models.configuration import Conf

class ModelBase(object):
  def __init__(self, dataset='mnist', model_name='abstract_model'):
    self.dataset = dataset
    self.model_name = model_name
    self.cache_dir = Conf.get('cache_dir')
    cache_file = '%s_%s' % (dataset, model_name)
    self.weights_file_name = os.path.join(self.cache_dir, cache_file)

    self.model = None

    self.load_weights_flag = False

    self.x_train = None
    self.y_train = None
    self.x_test = None
    self.y_test = None
    self.x_val = None
    self.y_val = None

    self.num_epochs = 10
    self.l1 = 0.0
    self.l2 = 0.0
    self.dropout_rate = 0.0

    self.layer_ids = []
    self.layer_prune_rates = []

    self.total_weights = 0

  def set_model(self, model, layer_ids, layer_prune_rates):
    self.model = model
    self.layer_ids = layer_ids
    self.layer_prune_rates = layer_prune_rates
    self.total_weights = model.count_params()

  def set_training_params(self, args):
    print args
    self.num_epochs = args.epochs
    self.l1 = args.l1
    self.l2 = args.l2
    self.dropout_rate = args.dropout_rate

  def set_data(self, x_train, y_train, x_test, y_test, x_val, y_val):
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.x_val = x_val
    self.y_val = y_val

  # optimizer can be RMSprop
  def compile_model(self, loss='categorical_crossentropy', optimizer='adadelta', metrics=None):
    if metrics is None:
      metrics=['accuracy']
    self.model.compile(loss=loss,
                         optimizer=optimizer,
                         metrics=metrics)

  def fit_model(self, batch_size=128, v=0):
    if False:
      print self.y_val.shape
      print self.x_val.shape
      print self.x_train.shape
    if False:#self.dataset is 'mnist':
      self.model.fit(self.x_train, self.y_train, batch_size=batch_size, \
                                 nb_epoch=self.num_epochs, verbose=v, \
                                 validation_split=(0.10))
    else:
      self.model.fit(self.x_train, self.y_train, batch_size=batch_size, \
                                 nb_epoch=self.num_epochs, verbose=v, \
                                 validation_data=(self.x_val, self.y_val))

  def eval_model(self, v=0):
    out = self.model.evaluate(self.x_val, self.y_val, verbose=v)
    return (1. - out[1])

  def get_layer_inputs(self, layer=-1):
    from keras import backend as K
    layer_func = K.function([self.model.layers[0].input],
                            [self.model.layers[layer].input])
    # [0] is because layer_func is a keras function, which take lists as inputs
    # and produce lists as outputs
    return layer_func([self.x_val])[0]

  def get_layer_outputs(self, layer=-1):
    from keras import backend as K
    layer_func = K.function([self.model.layers[0].input],
                            [self.model.layers[layer].output])
    # [0] is because layer_func is a keras function, which take lists as inputs
    # and produce lists as outputs
    return layer_func([self.x_val])[0]

  def get_activities(self, layer=-1):
    return get_layer_outputs(layer)

  def test_model(self, v=0):
    out = self.model.evaluate(self.x_test, self.y_test, verbose=v)
    return (1. - out[1])

  def save_weights(self, filename=None):
    if filename:
      fname = filename
    else:
      fname = self.weights_file_name

    print "FILENAME = ", fname
    self.model.save_weights(fname)

  def load_weights(self, fname=None, absolute=False):
    print (fname)
    if absolute:
      self.model.load_weights(fname)
    elif self.dataset is 'imagenet':
      self.model.load_weights(fname)
    elif fname:
      self.model.load_weights('%s/%s' % (self.cache_dir, fname))
    else:
      self.model.load_weights(self.weights_file_name)

  def get_layers(self):
    return self.model.layers

  def get_weights(self):
    return self.model.get_weights()

  def set_weights(self, w_list):
    return self.model.set_weights(w_list)

class IndirectModel(ModelBase):
  '''A model with indirect indices instead of a normal weight matrix.

  This model cannot be executed directly, as its weights are effectively
  meaningless.
  '''

  def __init__(self, *args, **kwargs):
    super(IndirectModel,self).__init__(*args, **kwargs)
    self.value_table = None

  _NIE_msg = 'Cannot execute this method on an indirect model. Convert to a regular model first.'
  def compile_model(self, *a, **k): raise NotImplementedError, self._NIE_msg
  def fit_model(self, *a, **k): raise NotImplementedError, self._NIE_msg
  def eval_model(self, *a, **k): raise NotImplementedError, self._NIE_msg
  def test_model(self, *a, **k): raise NotImplementedError, self._NIE_msg
  def save_weights(self, *a, **k): raise NotImplementedError, self._NIE_msg
  def load_weights(self, *a, **k): raise NotImplementedError, self._NIE_msg

  def get_values(self):
    '''Return the actual weight values shared by the model.'''
    return self.value_table
