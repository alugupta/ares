
mkdir -p logs
mkdir -p logs/cifar10
mkdir -p logs/mnist_fc
TRAINED_MODELS_DIR="/group/vlsiarch/ugupta/git/ares/ares/experiments/quantize/quantized_weights/"

SEED=0
FRATE=0.00002

fname="mnist_quantized_2_8_($FRATE)_$SEED"
THEANO_FLAGS='device=gpu' python bits.py -m mnist_fc -lw -qi 2 -qf 8 -ld_name $TRAINED_MODELS_DIR/mnist_fc_quantized_2_8 -frate $FRATE -seed $SEED | tee -a logs/mnist_fc/$fname
