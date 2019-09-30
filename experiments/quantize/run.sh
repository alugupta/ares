#!/bin/bash
mkdir -p /ares/quantized_weights

TRAINED_MODELS_DIR="/ares/experiments/train/trained_models"

THEANO_FLAGS='device=gpu' python /ares/experiments/quantize/quantize_net.py -m mnist_fc -lw -ld_name $TRAINED_MODELS_DIR/mnist_fc -qi 2 -qf 6 --cache /ares/cache/ --conf /ares/conf --results /ares/results -sw /ares/quantized_weights/mnist_fc

