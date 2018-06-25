# Sample run.sh
export PYTHONPATH=$(pwd)/:

mkdir -p cache
mkdir -p results

# Training MNIST
THEANO_FLAGS='device=gpu' python run_models.py -m mnist_fc -eps 10 -v  -to -c conf

# Training CiFar
#THEANO_FLAGS='device=gpu' python run_models.py -m cifar10_vgg -eps 10 -v  -to
