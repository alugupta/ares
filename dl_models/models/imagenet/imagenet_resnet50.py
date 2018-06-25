from keras.datasets             import mnist
from keras.models               import Sequential
from keras.layers.core          import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import ZeroPadding2D, Convolution2D
from keras.layers.pooling       import MaxPooling2D
from keras.regularizers         import l2
from keras.utils                import np_utils
from keras                      import backend

from keras.optimizers           import SGD

import operator as op

from dl_models.models.base import *

from dl_models.models.imagenet.imagenet_utils import *
from dl_models.models.model_configs import imagenetVGG16_params

import random
import sys

from keras.applications.resnet50 import ResNet50

class imagenetResNet50(ModelBase):
  def __init__(self):
    super(imagenetResNet50,self).__init__('imagenet','resnet50')

    # Conv layer params
    self.batch_size = 32

    self.num_train_blocks = 1
    self.num_val_blocks   = 1
    self.num_epochs       = 1

    # in dimensions
    self.nb_classes = 1000
    self.img_rows   = 224
    self.img_cols   = 224
    self.channels   = 3

    self.input_shape = (self.channels, self.img_rows, self.img_cols)

    self.param_layer_ids = []
    self.default_prune_factors = []

    self.l1 = 0.00001
    self.l2 = 5e-4

    # dir to store all the preprocessed files
    preprocessing_dir = imagenetResNet50_params['preprocessing_dir']

    print "Preprocessing dir = ", preprocessing_dir

    self.weights_file_name = preprocessing_dir + "imagenet_resnet_pretrained_th_weights"
    self.class_index_file  = preprocessing_dir + "imagenet_class_index.json"

    labels_dir          = preprocessing_dir + "labels/"
    self.misc_dir       = preprocessing_dir + "misc/"
    self.train_img_dir  = preprocessing_dir + "train_hkl_b256_b_256/"
    self.val_img_dir    = preprocessing_dir + "val_hkl_b256_b_256/"

    val_labels_filepath   = labels_dir + "val_labels.npy"
    train_labels_filepath = labels_dir + "train_labels.npy"

    labels = load_imagenet_labels(train_labels_filepath, val_labels_filepath)

    self.train_labels = labels[0]
    self.val_labels   = labels[1]

    self.train_map_filename      = self.misc_dir   + "train.txt"
    self.val_map_filename        = self.misc_dir   + "val.txt"

  def build_model(self,):
    # Recreate the Keras ResNet50 pre-built model here:
    model = ResNet50(weights="imagenet")

    self.set_model( model, self.param_layer_ids, self.default_prune_factors )

  def eval_model(self, v=0):
    # Run inference on a single image defined by the img_path
    mini_batch_size = self.batch_size

    total_imgs    = 0
    top_1_correct = 0
    top_5_correct = 0

    for batch_id in range(self.num_val_blocks):
      X_val, Y_val, datagen = get_image_batch( self.val_img_dir, \
                                               self.val_labels,  \
                                               256,         \
                                               batch_id)

      imagenet_map = imagenet_mapping(self.class_index_file)

      mini_batches = 0

      predictions = [ ]

      for b in range(256 / mini_batch_size):
          b_start = b * mini_batch_size
          p = self.model.predict_on_batch(X_val[b_start : b_start + mini_batch_size])
          predictions.append(p)

      predictions = [i for sublist in predictions for i in sublist]
      predictions = np.array(predictions)
      decoded     = decode_predictions(predictions)

      for i in range(len(predictions)):
          total_imgs += 1

          golden = np.argmax(Y_val[i])
          top_1_correct += ( decoded[i][0][0] == imagenet_map[golden])

          decoded_top5   = [ ]

          for j in range(5):
            decoded_top5.append(decoded[i][j][0])

          decoded_top5   = map(str, decoded_top5)
          top_5_correct += (str(imagenet_map[golden]) in decoded_top5)

    top_1_correct = float(top_1_correct) / float(total_imgs)
    top_5_correct = float(top_5_correct) / float(total_imgs)

    print "ImageNet Accuracy top1 = ", top_1_correct, " top5 = ",  top_5_correct
    return 1 - top_1_correct

  def fit_model(self, batch_size=32, v=0):
    loss_per_epoch  = [ ]

    for e in range(self.num_epochs):
        print 'Epoch', e
        # Shuffle blocks on disk
        shuffled_batch_ids = range(self.num_train_blocks)
        random.shuffle(shuffled_batch_ids)

        print "Shuffled Batch Ids = ", shuffled_batch_ids

        for block_id in shuffled_batch_ids:
          mini_batches = 0
          print "Random Shuffled Batch Id", block_id

          X_train, Y_train, datagen = get_image_batch( self.train_img_dir, \
                                                       self.train_labels, \
                                                       256, \
                                                       block_id)
          # By default flow shuffles the images
          loss = self.model.fit_generator(datagen.flow(X_train, Y_train,
              batch_size=self.batch_size), samples_per_epoch=len(X_train),
              nb_epoch = 1)

        loss_per_epoch.append(loss)

    return loss_per_epoch

  # overriding compile_model from base as loss should be mse and
  # optimizer sgd (work slightly better)
  def compile_model(self, loss='categorical_crossentropy', optimizer='sgd', metrics=None):
    if metrics is None:
      metrics=['accuracy']

    if optimizer is 'sgd':
      print "Using ResNet50 specific  optimizer SGD"
      learning_rate = imagenetResNet50_params['learning_rate']
      print "Learning rate = ", learning_rate
      opt = SGD(lr = learning_rate, decay = 1e-6, momentum=0.9, nesterov=True)
    else:
      sys.exit()

    self.model.compile(loss=loss,
                       optimizer=opt,
                       metrics=metrics)

  def load_dataset(self, ):
    return

  def test_model(self, v=0):
    return self.eval_model(v=v)

