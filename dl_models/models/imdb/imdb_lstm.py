from keras.datasets                    import imdb
from keras.models                      import Sequential
from keras.layers.core                 import Dense, Activation
from keras.layers.recurrent            import LSTM
from keras.layers.embeddings           import Embedding
from keras.layers.advanced_activations import ThresholdedReLU as TReLU
from keras.regularizers                import l2
from keras.utils                       import np_utils
from keras                             import backend


import operator as op

from keras.preprocessing import sequence
from dl_models.models.base import *

class imdbLSTM(ModelBase):
  def __init__(self):
    super(imdbLSTM,self).__init__('imdb','lstm')


    self.batch_size   = 32
    self.max_features = 20000
    self.max_len      = 80

    # in dimensions
    self.nb_classes = 1
    self.img_rows = 28
    self.img_cols = 28

    self.input_shape = self.img_rows * self.img_cols
    self.param_layer_ids       = ['Dense', 'Bias', 'Dense', 'Bias', 'Dense', 'Bias']
    self.default_prune_factors = ['0.001', '0.001','0.001', '0.001','0.001', '0.001']

    self.l1 = 0.00001
    self.l2 = 0.00001

    self.relu_threshold = 0.0 # Default off

  def load_dataset(self, ):
    (training, testing) = imdb.load_data(nb_words=self.max_features)
    (x_train, y_train)  = training
    (x_test, y_test)    = testing

    x_train = sequence.pad_sequences(x_train, maxlen=self.max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=self.max_len)

    self.set_data(x_train, y_train, x_test, y_test, x_test, y_test)

  def build_model(self,):
    model = Sequential()

    model.add(Embedding(self.max_features, 128, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    self.set_model( model, self.param_layer_ids, self.default_prune_factors )

  def eval_model(self, v=0):
    score, acc = self.model.evaluate(self.x_val, self.y_val, verbose=v, batch_size=self.batch_size)

    print('Test score:', score)
    print('Test accuracy:', acc)

    return (1. - acc)

  def compile_model(self, loss='binary_crossentropy', optimizer='RMSprop', metrics=None):
    if metrics is None:
      metrics=['accuracy']
    self.model.compile(loss='binary_crossentropy',
                         optimizer='RMSprop',
                         metrics=metrics)

