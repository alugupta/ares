'''Models provided for experimentation.'''

# MNIST Models
from mnist.mnist_fc          import mnistFC
from mnist.mnist_lenet5      import mnistLenet5

# IMDB Models
from imdb.imdb_rnn          import imdbRNN
from imdb.imdb_gru          import imdbGRU
from imdb.imdb_gru_bidir    import imdbGRUBidir
from imdb.imdb_lstm         import imdbLSTM

# SVHN Models
from svhn.svhn_lenet5       import svhnLenet5

# Imagenet Models
from imagenet.imagenet_vgg16    import imagenetVGG16
from imagenet.imagenet_resnet50 import imagenetResNet50

# Cifar10 Models
from cifar10.cifar10_vgg   import cifar10VGG

# TIDigits Models
from tidigits.tidigits_gru  import tidigitsGRU
from tidigits.tidigits_lstm import tidigitsLSTM
from tidigits.tidigits_rnn  import tidigitsRNN

from base              import ModelBase, IndirectModel
