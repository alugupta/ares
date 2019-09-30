#!/usr/bin/env python

import sys
import argparse

from dl_models.models.base import *

from dl_models.models import mnistFC
from dl_models.models import mnistLenet5
from dl_models.models import svhnLenet5
from dl_models.models import imagenetVGG16
from dl_models.models import imagenetResNet50
from dl_models.models import imagenetInceptionv3
from dl_models.models import cifar10VGG
from dl_models.models import tidigitsGRU
from dl_models.models import tidigitsRNN
from dl_models.models import tidigitsLSTM
from dl_models.models import cifar10alexnet


from dl_models.transform import SummarizeSparsity
from dl_models.transform import Retraining

model_class_map = {
                   'mnist_lenet5'      : mnistLenet5,
                   'mnist_fc'          : mnistFC,
                   'svhn_lenet5'       : svhnLenet5,
                   'imagenet_vgg16'    : imagenetVGG16,
                   'imagenet_resnet50' : imagenetResNet50,
                   'cifar10_vgg'       : cifar10VGG,
                   'tidigits_gru'      : tidigitsGRU,
                   'tidigits_rnn'      : tidigitsRNN,
                   'tidigits_lstm'      : tidigitsLSTM,
                   'imagenet_inceptionv3' : imagenetInceptionv3,
                   'cifar10_alexnet'   : cifar10alexnet,
                  }

def cli():
  # Default message.
  parser = argparse.ArgumentParser(description='Ares: A framework for quantifying the resilience of deep neural networks', \
                                        epilog='Configure you experiment from the command line.')

  parser.add_argument('-m', '--model', required=True, type=str, \
                           help='Pick a model to run. Models listed in models/model_config.py')

  parser.add_argument('-to', '--train_once', action='store_true', help='Train the model.')

  parser.add_argument('-sw', '--save_weights', action='store_true', help='Save weights to cache.')
  parser.add_argument('-lw', '--load_weights', action='store_true', help='Load saved weights from cache.')
  parser.add_argument('-ld_rt', '--load_pruned_value', default='0.0', type=float, \
                           help='Specifiy the saved pruned weights to use.')
  parser.add_argument('-ld_name', '--weight_name', default=None, type=str, \
                           help='Specifiy the weights to use.')

  parser.add_argument('-l2', '--l2',      default='0.0', type=float, help='Set l2 reg penalty.')
  parser.add_argument('-dr_rt', '--dropout_rate', default='0.35', type=float, help='Dropout rate for appropriate layers.')
  parser.add_argument('-eps', '--epochs', default=15, type=int, help='Num of training epochs.')

  parser.add_argument('-pwdist', '--plot_weight_distributions', action='store_true', help='Plot each layer weight distributions.')
  parser.add_argument('-c','--configuration', type=str, default=None, help='Specify a configuration file.')
  parser.add_argument('-v','--verbose', action='store_true', default=False, help='Enable verbose training output')

  parser.add_argument('-cache','--cache', type=str, default=None, help='Specify a cache dir.')
  parser.add_argument('-results','--results', type=str, default=None, help='Specify results dir.')
  parser.add_argument('-seed', '--seed', default=1337, type=int, help='Random seed for reproducible training runs')

  args = parser.parse_args()
  return args

def load_and_build(model, args):
  # build the model
  model.load_dataset()
  model.set_training_params(args)
  model.build_model()
  model.compile_model()

  if args.load_weights:
    model.load_weights(args.weight_name)
    print('Testing loaded weights:')
    err = model.eval_model()
    print(('Error of loaded weights--', err))

  #model.model.summary()

def train_and_save(model, args):
  # if we didn't load weights, we can train a new DNN.
  if args.train_once:
    model.fit_model(v=args.verbose)
    model.eval_model()

  if args.save_weights:
    model.save_weights()

  if args.plot_weight_distributions:
    model.plot_distributions()

  error = model.eval_model()
  print(('Validation error: %f' % error))
  error = model.test_model()
  print(('Test error: %f' % error))

def process_model(model, args):
  # Get weights, setup model and build functions
  load_and_build(model, args)
  # train and sweep if necessary
  if (not args.load_weights):
    train_and_save(model, args)

def config_setup(args):
  if args.configuration is not None:
    print("[Conf] Using configuration from:" + args.configuration)
    Conf.load(Conf.find_config(args.configuration))
  else:
    print("[Conf] Using default environment configuration")
    Conf.set_env_default()

  if args.cache is not None:
    Conf.set('cache', args.cache)

  if args.results is not None:
    Conf.set('results', args.results)

if __name__ == '__main__':
  args = cli()
  np.random.seed(args.seed)
  config_setup(args)
  model_name = args.model
  ModelClass = model_class_map[model_name]
  model = ModelClass()
  print('Experimenting with model: %s' % model.model_name)
  process_model(model, args)
