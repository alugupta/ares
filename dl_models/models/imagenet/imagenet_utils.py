from keras.preprocessing       import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16  import preprocess_input, decode_predictions
from keras.utils.np_utils      import to_categorical

import numpy as np
import pickle, hickle
import sys
import glob
import json
import scipy

# for colored outputs (warning messages and such)
class bcolors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'

''' Loads train.npy, val.npy and shuffled_train_filenames.npy'''
def load_imagenet_labels(train_label_filename,
                         val_label_filename):

    train_labels             = np.load(train_label_filename)
    val_labels               = np.load(val_label_filename)

    return (train_labels, val_labels)

def load_pickled_imgs(img_dir, batchid = None):
    img_pickles = sorted(glob.glob(img_dir + "*.hkl"))
    batches     = [ ]

    if batchid is not None:
        # load a specific batch
        img_pickles = [ img_pickles [ batchid ] ]
    else:
        print bcolors.WARNING + \
              "Be careful! You are starting to load the entire "  \
              + "dataset. This could be very big for all of " \
              + "ImageNet. Please specify a specific batchid " \
              + "if possible " \
              + bcolors.ENDC

    for img_pickle in img_pickles:
        # Should be array of shape (3, img_size, img_size, batch_size)
        # first img  = batch[ :, :, :, 0]
        #   where img[0] would then be the first channel (e.g. R in RGB)
        #         img[1] would then be the first channel (e.g. G in RGB)
        #         img[2] would then be the first channel (e.g. B in RGB)
        batch = hickle.load ( img_pickle )

        batch = np.swapaxes( np.array(batch), 1, 2)
        batch = np.swapaxes( batch, 2, 3)

        batches.append(batch)

    if batchid is None:
        # should have loaded entire image dataset
        assert (len(batches) == len(img_pickles))

    return batches

def imagenet_mapping(class_index_file):
  imagenet_map = { }
  with open(class_index_file) as imagenet_class:
      d = json.load(imagenet_class)

      for i in range(1000):
          imagenet_map[ i ] = d[str(i)][0]
  return imagenet_map

def resize_img(img, img_size):
    target_shape = (img_size, img_size, 3)

    #assert img.dtype == 'uint8'
    # assert False

    img = np.reshape(img, (img.shape[1], img.shape[2], img.shape[0]) )
    img = scipy.misc.imresize(img, target_shape)
    img = np.reshape(img, (img.shape[2], img.shape[1], img.shape[0]) )

    return img

def get_image_batch(img_dir, labels, batch_size, batch_id, img_size = None):

    img_batches = load_pickled_imgs(img_dir, batch_id)

    # Resize images (original size is 256x256)
    if img_size is not None:

        print bcolors.FAIL + \
                "SHOULD NOT BE RESIZING IMAGE ON THE FLY. THIS SHOULD HAVE BEEN \
                DONE IN PRE-PROCESSING!" + \
              bcolors.ENDC
        X_orig = img_batches[0]

        X = [ ]
        for x in X_orig:
            X.append(resize_img(x, img_size))

        X = np.array(X)
    else:
        X = img_batches[0]

    Y = labels [batch_size * batch_id : batch_size * (batch_id + 1) ]

    # assert that the number of images in X_train and Y_train is the same
    assert (X.shape[0] == Y.shape[0])

    # Can change the parameters later as we wish
    datagen = ImageDataGenerator()

    # 1000 categories since Imagenet's Wordnet Sysid's are from 0 to 999
    Y = to_categorical(Y, 1000)

    datagen.fit(X)

    return X, Y, datagen
