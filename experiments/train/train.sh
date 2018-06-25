TRAINED_MODELS_DIR="/group/vlsiarch/ugupta/git/ares/ares/experiments/train/trained_models/"

mkdir -p $TRAINED_MODELS_DIR

# Training MNIST-FC Model (LeNetFC)
THEANO_FLAGS='device=gpu' python train.py -m mnist_fc -eps 10 -v  -to -sw_name  $TRAINED_MODELS_DIR/mnist_fc

# Training CiFar10 VGG model
#THEANO_FLAGS='device=gpu' python train.py -m cifar10_vgg -eps 30 -v  -to -sw_name  $TRAINED_MODELS_DIR/cifar10_vgg -lr 0.0005 -seed 1
