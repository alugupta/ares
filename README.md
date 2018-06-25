# Ares: A framework for quantifying the resilience of deep neural networks

## Models
This repository supports the training, optimization, and evaluation of the following models:

1. MNIST-LeNet300-100 ( Three FC layers, MNIST dataset                         )
2. MNIST-LeNet5       ( Two convolutional layers, two FC layers, MNIST dataset )
3. TIDIGITS-X         ( Simple RNN, GRU, and LSTM networks, TIDIGITS dataset   )
4. CiFar10-VGG        ( VGG-esque convolutional network, CiFar10 dataset       )
5. ImageNet-VGG16     ( VGG16 convolutional network, ImageNet dataset          )
6. ImageNet-ResNet50  ( ResNet50 convolutional network, ImageNet dataset       )

Follow the structure in `ares/dl_models/models/` to modify the model architectures or build additional models.

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
[TIDIGITs](https://catalog.ldc.upenn.edu/ldc93s10) and [ImageNet](http://www.image-net.org/) must be downloaded and pre-processed separately, however pre-trained models for ImageNet (e.g., VGG16, ResNet50) are available through Keras as well.

### Quantization
To quantize models, run [`ares/experiments/quantize/run.sh`](./experiments/quantize/run.sh).

The [quantization transform](./dl-models/transform/quantize.py) emulates fixed-point datatypes for weights. Note that activations and arithmetic operations still have full-precision.

### Evaluation
After training and quantizing models, `ares/experiments/eval/eval.sh` can be used to evaluate the models on the validation and test sets.

### Fault-injection

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
