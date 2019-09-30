
mkdir -p logs
mkdir -p logs/cifar10
mkdir -p logs/mnist_fc
TRAINED_MODELS_DIR="/ares/quantized_weights"

SEED=0
FRATE=0.00005

fname="mnist_quantized_2_8_($FRATE)_$SEED"
THEANO_FLAGS='device=gpu' python /ares/experiments/bits/bits.py -c /ares/conf -m mnist_fc -lw -qi 2 -qf 6 -ld_name $TRAINED_MODELS_DIR/mnist_fc_quantized_2_6 -frate $FRATE -seed $SEED | tee -a logs/mnist_fc/$fname
