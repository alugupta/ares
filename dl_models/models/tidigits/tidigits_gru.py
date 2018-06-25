from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.advanced_activations import ThresholdedReLU as TReLU
from keras.regularizers import l2
from keras.utils import np_utils
from keras import backend

from keras import optimizers

from keras.layers import Flatten

from keras.layers.recurrent import GRU

import operator as op

from dl_models.models.base import *
import h5py
import sys
import numpy as np

class tidigitsGRU(ModelBase):
  def __init__(self):
    super(tidigitsGRU,self).__init__('tidigits','gru')

    # Conv layer params
    self.batch_size = 128

    # in dimensions
    self.nb_classes = 10

    self.param_layer_ids       = ['Dense', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001', '0.001','0.001', '0.001','0.001', '0.001']

    self.l1 = 0.00001
    self.l2 = 0.00001

    self.relu_threshold = 0.0 # Default off

    self.timesteps = 254
    self.features  = 39

  def build_model(self,):
    model = Sequential()

    model.add(GRU(200, return_sequences = True, input_shape = (self.timesteps, self.features)))

    model.add(Flatten())

    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    self.set_model( model, self.param_layer_ids, self.default_prune_factors )

  def numpyify(self, data):
    shape = data.shape
    numpy_data = np.array(data)

    return numpy_data

  # Converts single element of tidigits targets (y_train and y_test) from type
  # string to type int
  def convert_tidigits(self, x):
    if x == b'Z' or x == b'O':
      return 0
    else:
      return int(x)

  # Read in tidigits.hdf5. Assumes dataset is one directory up
  def read_tidigits(self, fname):
    tidigits = h5py.File(fname, 'r')
    print (tidigits)

    print (tidigits.keys())

    for i, key in enumerate(tidigits.keys()):
      if i == 0:
        x_train_key = key
      elif i == 1:
        x_test_key =  key
      elif i == 2:
        y_train_key =  key
      elif i == 3:
        y_test_key =  key


    # Training Set
    x_train = tidigits[x_train_key]
    y_train = tidigits[y_train_key]

    # Testing Set
    x_test = tidigits[x_test_key]
    y_test = tidigits[y_test_key]

    # Original labels for test and training set are in string format. We must
    # convert them into integers as strings cannot be made into tensor types
    y_test  = [self.convert_tidigits(s) for s in y_test]
    y_train = [self.convert_tidigits(s) for s in y_train]

    train = (self.numpyify(x_train), y_train)
    test  = (self.numpyify(x_test) , y_test)

    return tidigits, train, test

  def preprocess_categorical(self, ys, yclasses=2, dtype='float32'):
    new_ys = []
    for y in ys:
      # Convert categorical labels to binary (1-hot) encoding
      y = np_utils.to_categorical(y, yclasses)
      new_ys.append(y)
    return np.array(new_ys)

  def load_dataset(self, ):
    try:
      tidigits, ti_train, ti_test = self.read_tidigits('/group/brooks/dl_models/tidigits/data/tidigits.hdf5')
    except:
      print "Could not locate TIDIGITS dataset!"
      print "TIDIGITS dataset must be purchased and pre-processed into hdf5 file"
      sys.exit(1)

    x_train, y_train  = ti_train
    x_test, y_test    = ti_test

    y_train = self.preprocess_categorical(y_train, 10)
    y_test  = self.preprocess_categorical(y_test, 10)

    y_train = np.squeeze(y_train, 1)
    y_test = np.squeeze(y_test, 1)

    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')

    self.set_data(x_train, y_train, x_test, y_test, x_test, y_test)

    print x_train.shape
    print y_train.shape
    print x_test.shape
    print y_test.shape

  def compile_model(self, loss='categorical_crossentropy', optimizer='sgd', metrics=None):
    self.model.compile(loss=loss,
                         optimizer=optimizers.SGD(lr = 0.01, momentum = 0, decay = 0, nesterov=False),
                         metrics=['accuracy'])
