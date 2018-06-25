#!/usr/bin/env python


import sys
import argparse

import dl_models

from dl_models.models.model_configs import *
from dl_models.models.base import *

from dl_models.models import mnistFC
from dl_models.models import mnistLenet5
from dl_models.models import svhnLenet5
from dl_models.models import imagenetVGG16
from dl_models.models import imagenetResNet50
from dl_models.models import cifar10VGG
from dl_models.models import tidigitsGRU

from dl_models.transform import SummarizeSparsity
from dl_models.transform import Retraining
import cProfile

model_class_map = {
                   'mnist_lenet5'      : mnistLenet5,
                   'mnist_fc'          : mnistFC,
                   'svhn_lenet5'       : svhnLenet5,
                   'cifar10_vgg'       : cifar10VGG,
                   'imagenet_vgg16'    : imagenetVGG16,
                   'imagenet_resnet50' : imagenetResNet50,
                   'tidigits_gru'      : tidigitsGRU,
                  }

def cli():
  # Default message.
  parser = argparse.ArgumentParser(description='Harvard VLSI-ARCH deep learning model framework', \
                                        epilog='Configure you experiment from the command line.')

  parser.add_argument('-m', '--model', required=True, type=str, \
                           help='Pick a model to run. Models listed in models/model_config.py')

  parser.add_argument('-lw', '--load_weights', action='store_true', help='Load saved weights from cache.')
  parser.add_argument('-ld_name', '--weight_name', default=None, type=str, \
                           help='Specifiy the weights to use.')

  parser.add_argument('-pwdist', '--plot_weight_distributions', action='store_true', help='Plot each layer weight distributions.')
  parser.add_argument('-c','--configuration', type=str, default=None, help='Specify a configuration file.')
  parser.add_argument('-v','--verbose', action='store_true', default=False, help='Enable verbose training output')

  parser.add_argument('-cache','--cache', type=str, default=None, help='Specify a cache dir.')
  parser.add_argument('-results','--results', type=str, default=None, help='Specify results dir.')

  args = parser.parse_args()
  return args

def load_and_build(model, args):
  # build the model and Theano functions.
  model.load_dataset()

  model.build_model()
  model.compile_model()

  if args.load_weights:
    model.load_weights(args.weight_name, absolute=True)
    print('Testing loaded weights')
    err = model.eval_model()
    print('Validation error ', err)
    err = model.test_model()
    print('Test error ', err)

def process_model(model, args):
  # Get weights, setup model and build functions
  load_and_build(model, args)

def config_setup(args):
  if args.configuration is not None:
    print "[Conf] Using configuration from:" + args.configuration
    Conf.load(Conf.find_config(args.configuration))
  else:
    print "[Conf] Using default environment configuration"
    Conf.set_env_default()

  if args.cache is not None:
    Conf.set('cache', args.cache)

  if args.results is not None:
    Conf.set('results', args.results)

if __name__ == '__main__':
  args = cli()
  config_setup(args)
  model_name = args.model
  ModelClass = model_class_map[model_name]
  model = ModelClass()
  print 'Training model: %s' % model.model_name
  process_model(model, args)
