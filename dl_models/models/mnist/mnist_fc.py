from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.advanced_activations import ThresholdedReLU as TReLU
from keras.regularizers import l2
from keras.utils import np_utils
from keras import backend

import operator as op

from dl_models.models.base import *

class mnistFC(ModelBase):
  def __init__(self):
    super(mnistFC,self).__init__('mnist','fc')

    # Conv layer params
    self.batch_size = 128

    # in dimensions
    self.nb_classes = 10
    self.img_rows = 28
    self.img_cols = 28

    self.input_shape = self.img_rows * self.img_cols
    self.param_layer_ids       = ['Dense', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001', '0.001','0.001', '0.001','0.001', '0.001']

    self.l1 = 0.00001
    self.l2 = 0.00001

    self.relu_threshold = 0.0 # Default off

  def preprocess_categorical(self, xs, ys, yclasses=2, dtype='float32'):
    new_xs = []
    for x in xs:
      # Flatten, convert dtype, and standardize
      x = x.reshape((x.shape[0],reduce(op.mul,x.shape[1:])))
      x = x.astype('float32')
      (mu,sig) = (np.mean(x,axis=1,keepdims=True), np.std(x,axis=1,keepdims=True))
      x = (x-mu)/sig
      new_xs.append(x)
    new_ys = []
    for y in ys:
      # Convert categorical labels to binary (1-hot) encoding
      y = np_utils.to_categorical(y, yclasses)
      new_ys.append(y)
    return zip(new_xs, new_ys)

  def load_dataset(self, ):
    (training, testing) = mnist.load_data()
    ((x_train, y_train),(x_test, y_test)) = self.preprocess_categorical(*zip(training, testing), yclasses=10)

    if backend.image_dim_ordering() == 'th':
      print('Using TH')
      x_train = x_train.reshape(x_train.shape[0], self.input_shape)
      x_test = x_test.reshape(x_test.shape[0], self.input_shape)
    else:
      print('Using TF')
      x_train = x_train.reshape(x_train.shape[0], self.input_shape)
      x_test = x_test.reshape(x_test.shape[0], self.input_shape)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    self.set_data(x_train, y_train, x_test, y_test, x_test, y_test)

  def build_model(self,):
    model = Sequential()

    model.add(Dense(300, input_shape=(784,), W_regularizer=l2(self.l2)))
    model.add(Activation('relu'))
    model.add(Dense(100, W_regularizer=l2(self.l2)))

    model.add(Activation('relu'))
    model.add(Dense(self.nb_classes, W_regularizer=l2(self.l2)))

    model.add(Activation('softmax'))
    self.set_model( model, self.param_layer_ids, self.default_prune_factors )

