TRAINED_MODELS_DIR="/group/vlsiarch/ugupta/git/ares/ares/experiments/train/trained_models/"

# Evaluate CiFar
#THEANO_FLAGS='device=gpu' python eval.py -m mnist_fc -v -lw -ld_name $TRAINED_MODELS_DIR/mnist_fc


PRETRAINED_WEIGHTS="/group/vlsiarch/ugupta/git/ares/ares-pretrained-models/"
# Testing pre-trained weights
#THEANO_FLAGS='device=gpu' python eval.py -m mnist_fc -v -lw -ld_name $PRETRAINED_WEIGHTS/mnist_fc_quantized_3_13
#THEANO_FLAGS='device=gpu' python eval.py -m mnist_fc -v -lw -ld_name $PRETRAINED_WEIGHTS/mnist_fc_quantized_2_6
#
#THEANO_FLAGS='device=gpu' python eval.py -m mnist_lenet5 -v -lw -ld_name $PRETRAINED_WEIGHTS/mnist_lenet5_quantized_3_13
#THEANO_FLAGS='device=gpu' python eval.py -m mnist_lenet5 -v -lw -ld_name $PRETRAINED_WEIGHTS/mnist_lenet5_quantized_2_8
#
#THEANO_FLAGS='device=gpu' python eval.py -m cifar10_vgg -v -lw -ld_name $PRETRAINED_WEIGHTS/cifar10_alexnet_quantized_3_13
#THEANO_FLAGS='device=gpu' python eval.py -m cifar10_vgg -v -lw -ld_name $PRETRAINED_WEIGHTS/cifar10_alexnet_quantized_2_10

#THEANO_FLAGS='device=gpu' python eval.py -m imagenet_vgg16 -v -lw -ld_name $PRETRAINED_WEIGHTS/imagenet_vgg16_quantized_3_13
#THEANO_FLAGS='device=gpu' python eval.py -m imagenet_vgg16 -v -lw -ld_name $PRETRAINED_WEIGHTS/imagenet_vgg16_quantized_2_10

THEANO_FLAGS='device=gpu' python eval.py -m tidigits_gru -v -lw -ld_name $PRETRAINED_WEIGHTS/tidigits_gru_quantized_3_13
THEANO_FLAGS='device=gpu' python eval.py -m tidigits_gru -v -lw -ld_name $PRETRAINED_WEIGHTS/tidigits_gru_quantized_2_12
