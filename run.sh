# Sample run.sh
export PYTHONPATH=$(pwd)/:

mkdir -p cache
mkdir -p results

# Training MNIST
python3 /ares/run_models.py -m mnist_fc -eps 10 -v  -to -c /ares/conf -sw

# Training CiFar
#python /ares/run_models.py -m cifar10_vgg -eps 10 -v  -to -c /ares/conf

#Training SVHN
#python /ares/run_models.py -m imagenet_resnet50 -eps 10 -v  -to -c /ares/conf
