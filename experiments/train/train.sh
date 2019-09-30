TRAINED_MODELS_DIR="/ares/experiments/train/trained_models/"

mkdir -p $TRAINED_MODELS_DIR

# Training Model

python /ares/experiments/train/train.py -c /ares/conf -m mnist_fc -eps 10 -v  -to -sw_name  $TRAINED_MODELS_DIR/mnist_fc

