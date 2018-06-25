from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend

from dl_models.models.model_configs import svhn_lenet5_params

import operator as op

import numpy as np

from dl_models.models.base import *
import sys

class svhnLenet5(ModelBase):
  def __init__(self):
    super(svhnLenet5,self).__init__('svhn','lenet5')

    # Conv layer params
    self.batch_size = 128
    self.nb_epoch   = 20

    # in dimensions
    self.nb_classes  = 10
    self.img_rows    = 32
    self.img_cols    = 32
    self.channels    = 3
    self.nb_filters  = 32
    self.pool_size   = (2, 2)
    self.kernel_size = (3, 3)

    self.data_dir = svhn_lenet5_params['preprocessing_dir']

    self.input_shape = (self.channels, self.img_rows, self.img_cols)
    self.layer_ids             = ['Conv', 'Bias', 'Conv', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001','0.001','0.001','0.001','0.001', '0.001','0.001', '0.001']

  def load_dataset(self, ):
    X_train = np.load(self.data_dir + "X_train.npy")
    Y_train = np.load(self.data_dir + "Y_train.npy")

    X_test = np.load(self.data_dir + "X_test.npy")
    Y_test = np.load(self.data_dir + "Y_test.npy")

    Y_train = Y_train.flatten()
    Y_test  = Y_test.flatten()

    # Labels for SVHN are actually 1 to 10 so we we subtract one so that 10 classes are 0 through 9
    Y_train = map(lambda y: y - 1, Y_train)
    Y_test  = map(lambda y: y - 1, Y_test)

    Y_train = np_utils.to_categorical(Y_train, 10)
    Y_test  = np_utils.to_categorical(Y_test, 10)

    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')

    print('X_train shape is:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    self.set_data(X_train, Y_train, X_test, Y_test, X_test, Y_test)

  def build_model(self,):
    model = Sequential()

    model.add(Conv2D(self.nb_filters, (self.kernel_size[0],
        self.kernel_size[1]), input_shape=self.input_shape, data_format="channels_first"))

    model.add(Activation('relu'))
    model.add(Conv2D(self.nb_filters, (self.kernel_size[0], self.kernel_size[1]), data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=self.pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(self.nb_classes))
    model.add(Activation('softmax'))

    self.set_model( model, self.layer_ids, self.default_prune_factors )

