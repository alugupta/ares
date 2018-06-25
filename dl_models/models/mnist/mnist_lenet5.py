from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend
import operator as op

from dl_models.models.base import *

class mnistLenet5(ModelBase):
  def __init__(self):
    super(mnistLenet5,self).__init__('mnist','fc')

    # Conv layer params
    self.batch_size = 128
    self.nb_epoch = 3

    # in dimensions
    self.nb_classes = 10
    self.img_rows = 28
    self.img_cols = 28
    self.nb_filters = 32
    self.pool_size = (2, 2)
    self.kernel_size = (3, 3)

    self.input_shape = self.img_rows * self.img_cols
    self.layer_ids             = ['Conv', 'Bias', 'Conv', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001','0.001','0.001','0.001','0.001', '0.001','0.001', '0.001']

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
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    (training, testing) = mnist.load_data()
    ((x_train, y_train),(x_test, y_test)) = self.preprocess_categorical(*zip(training, testing), yclasses=10)

    if backend.image_dim_ordering() == 'th':
      print('Using TH')
      x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
      x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
      self.input_shape = (1, self.img_rows, self.img_cols)
    else:
      print('Using TF')
      x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
      x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
      self.input_shape = (self.img_rows, self.img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    print('x_train shape is:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    self.set_data(x_train, y_train, x_test, y_test, x_test, y_test)

  def build_model(self,):
    model = Sequential()

    model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1],
                              border_mode='valid', input_shape=self.input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1]))
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

