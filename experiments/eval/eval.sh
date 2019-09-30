
PRETRAINED_WEIGHTS="/ares/quantized_weights/"
# Testing pre-trained weights
python3 /ares/experiments/eval/eval.py -m mnist_fc -v  -c /ares/conf -lw -ld_name $PRETRAINED_WEIGHTS/mnist_fc_quantized_2_6

