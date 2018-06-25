#!/bin/bash
mkdir -p quantized_weights

TRAINED_MODELS_DIR="/group/vlsiarch/ugupta/git/ares/ares/experiments/train/trained_models/"
mkdir -p quantized_weights

THEANO_FLAGS='device=gpu' python quantize_net.py -m mnist_fc -lw -ld_name $TRAINED_MODELS_DIR/mnist_fc -qi 2 -qf 8 --cache ../../cache/ --results ../../results_dir -sw quantized_weights/mnist_fc

#THEANO_FLAGS='device=gpu' python quantize_net.py -m cifar10_vgg -lw -ld_name $TRAINED_MODELS_DIR/cifar10_vgg -qi 2 -qf 8 --cache ../../cache/ --results ../../results_dir -sw quantized_weights/cifar10
