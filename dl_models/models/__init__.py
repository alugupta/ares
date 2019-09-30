'''Models provided for experimentation.'''

# MNIST Models
from .mnist.mnist_fc          import mnistFC
from .mnist.mnist_lenet5      import mnistLenet5

# SVHN Models
from .svhn.svhn_lenet5       import svhnLenet5

# Imagenet Models
from .imagenet.imagenet_vgg16    import imagenetVGG16
from .imagenet.imagenet_resnet50 import imagenetResNet50
from .imagenet.imagenet_inceptionv3 import imagenetInceptionv3
# Cifar10 Models
from .cifar10.cifar10_vgg   import cifar10VGG
from .cifar10.cifar10_alexnet   import cifar10alexnet

# TIDigits Models
from .tidigits.tidigits_gru  import tidigitsGRU
from .tidigits.tidigits_lstm import tidigitsLSTM
from .tidigits.tidigits_rnn  import tidigitsRNN

from .base              import ModelBase, IndirectModel
