# Ares: A framework for quantifying the resilience of deep neural networks

## Models
This repository supports the training, optimization, and evaluation of the following models:

1. MNIST-LeNet300-100 ( Three FC layers, MNIST dataset                         )
2. MNIST-LeNet5       ( Two convolutional layers, two FC layers, MNIST dataset )
3. TIDIGITS-X         ( Simple RNN, GRU, and LSTM networks, TIDIGITS dataset   )
4. CiFar10-VGG        ( VGG-esque convolutional network, CiFar10 dataset       )
5. CiFar10-AlexNet
6. ImageNet-VGG16     ( VGG16 convolutional network, ImageNet dataset          )
7. ImageNet-ResNet50  ( ResNet50 convolutional network, ImageNet dataset       )
8. ImageNet-IncenptionV3
9. ImageNet-MobileNet

Follow the structure in [`ares/dl_models/models/`](./dl_models/models/) to modify the model architectures or build additional models.

## Getting started
To get started quickly, you can run simple training and testing examples using the various configurations in [`ares/run_models.py`](./run_models.py), and some examples are provided in [`ares/run.sh`](./run.sh).
You may need to edit the run_experiment.sh file to correctly configure paths to imagenet and other external data.
A docker script to create the correct path structure and run the tools is provided. Run [`ares/run_experiement.sh`](./run_experiment.sh)

## Ares Experiments
The experiments in [`ares/experiments`](./experiments) are provided to perform the fault-injection analyses found in Ares.
We provide examples for how to train, evaluate, quantize, and inject faults in models under [`ares/experiments/train`](./experiments/train), [`ares/experiments/eval`](./experiments/evaluate), [`ares/experiments/quantize`](./experiments/quantize), and [`ares/experiments/bits`](./experiments/bits).

Before running each of the experiments, add the Ares' root directory to your `PYTONPATH` as follows: `export PYTHONPATH=<path-to-ares>/:$PYTOHNPATH`.
In addition, for tidigits and imagenet, edit imagenet_base.py, run_experiment.py and/or tidigits_utils.py to make sure the data filepaths are correct. For imagenet, ares assumes the train and test set are stored one 256 batch per file, with the labels stored as a numpy array.

### Training
Examples of training models can be found in [`ares/experiments/train/train.sh`](./experiments/train/train.sh).

The MNIST and CiFAR datasets are made available through the Keras deep learning framework.
[TIDIGITs](https://catalog.ldc.upenn.edu/ldc93s10) and [ImageNet](http://www.image-net.org/) must be downloaded and pre-processed separately, however pre-trained models for ImageNet (e.g., VGG16, ResNet50) are available through Pytorch and will be automatically downloaded by the framework.

### Quantization
To quantize models, run [`ares/experiments/quantize/run.sh`](./experiments/quantize/run.sh).

The [quantization transform](./dl_models/transform/quantize.py) emulates fixed-point datatypes for weights. Note that activations and arithmetic operations still have full-precision.

### Evaluation
After training and quantizing models, `ares/experiments/eval/eval.sh` can be used to evaluate the models on the validation and test sets.

### Fault-injection
The [`ares/experiments/bits/bits.py`](./experiments/bits/bits.py) experiment implements the core fault-injection framework of Ares.
We illustrate an example of how to inject static persistent faults in the weights for the models.
The transform for random fault injection is implemented in [`ares/dl_models/transform/random_fault.py`](./dl_models/transform/random_fault.py).
Examples of running the fault injection framework can be found in [`ares/experiments/bits/run.sh`](./experiments/bits/run.sh).

Examples of how to inject faults into activations will be provided shortly. 
In the meantime, please contact us if you have questions regarding this. 

## Link to paper
Visit ACM's digital library to read the [full paper](https://dl.acm.org/citation.cfm?id=3195997).

## Citation
If you use `Ares`, please cite us:
```
   @inproceedings{Reagen:2018:AFQ:3195970.3195997,
   author = {Reagen, Brandon and Gupta, Udit and Pentecost, Lillian and Whatmough, Paul and Lee, Sae Kyu and Mulholland, Niamh and Brooks, David and Wei, Gu-Yeon},
   title = {Ares: A Framework for Quantifying the Resilience of Deep Neural Networks},
   booktitle = {Proceedings of the 55th Annual Design Automation Conference},
   series = {DAC '18},
   year = {2018},
   isbn = {978-1-4503-5700-5},
   location = {San Francisco, California},
   pages = {17:1--17:6},
   articleno = {17},
   numpages = {6},
   url = {http://doi.acm.org/10.1145/3195970.3195997},
   doi = {10.1145/3195970.3195997},
   acmid = {3195997},
   publisher = {ACM},
   address = {New York, NY, USA},}
   ```

## Contact Us
For any further questions please contact <harvard.vlsiarch.ares@gmail.com>
