# Ares: A framework for quantifying the resilience of deep neural networks

## Models
This repository supports the training, optimization, and evaluation of the following models:

1. MNIST-LeNet300-100 ( Three FC layers, MNIST dataset                         )
2. MNIST-LeNet5       ( Two convolutional layers, two FC layers, MNIST dataset )
3. TIDIGITS-X         ( Simple RNN, GRU, and LSTM networks, TIDIGITS dataset   )
4. CiFar10-VGG        ( VGG-esque convolutional network, CiFar10 dataset       )
5. ImageNet-VGG16     ( VGG16 convolutional network, ImageNet dataset          )
6. ImageNet-ResNet50  ( ResNet50 convolutional network, ImageNet dataset       )

## Getting started
To get started quickly, you can run simple training and testing examples using the various configurations in `ares/run_models.py`, and some examples are provided in `ares/run.sh`.
In order to run the script and experiments you will need to create `cache` and `results` directories (e.g., `ares/cache` and `ares/results`) where models weight and generated results (i.e., plots) are stored respectively.
Please add the appropriate paths to `ares/conf` as well.

## Ares Experiments
The experiments in `ares/experiments` are provided to perform the fault-injection analyses found in Ares.
We provide examples for how to train, evaluate, quantize, and inject faults in models under `ares/experiments/train`, `ares/experiments/evaluate`, `ares/experiments/quantize`, and `ares/experiments/bits`.

Before running each of the experiments, add the Ares' root directory to your `PYTONPATH` as follows: `export PYTHONPATH=<path-to-ares>/:$PYTOHNPATH`.

### Training
Examples of training models can be found in `ares/experiments/train/train.sh`.

The MNIST and CiFAR datasets are made available through the Keras deep learning framework.
TIDIGITs and ImageNet must be downloaded and pre-processed separately, however pre-trained models for ImageNet (e.g., VGG16, ResNet50) are available through Keras as well.

### Quantization
To quantize models, run `ares/experiments/quantize/run.sh`.

### Evaluation
After training and quantizing models, `ares/experiments/eval/eval.sh` can be used to evaluate the models on the validation and test sets.

### Fault-injection

