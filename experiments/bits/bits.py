import dl_models
import argparse
import sys
import cProfile
import hickle
import copy
import random

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
  parser = argparse.ArgumentParser(description='Bit level fault injection experiment', \
                                        epilog='Configure your experiment from the command line.')

  parser.add_argument('-m', '--model', required=True, type=str, \
                       help='Pick a model to run. Models listed in models/model_config.py')

  parser.add_argument('-lw', '--load_weights', action='store_true', help='Load saved weights from cache.')

  parser.add_argument('-ld_name', '--weight_name', default=None, type=str, \
                           help='Specifiy the weights to use.')

  parser.add_argument('-qi', '--qi', default=2, type=int, help='Integer bits for quantization')
  parser.add_argument('-qf', '--qf', default=6, type=int, help='Fractional bits for quantization')

  parser.add_argument('-seed', '--seed', default=0xdeadbeef, type=int, help='Random seed for bit-level fault injector')
  parser.add_argument('-frate', '--frate', default=0.0001, type=float, help='Fault Rate')

  parser.add_argument('-c','--configuration', type=str, default=None, help='Specify a configuration file.')
  parser.add_argument('-cache','--cache', type=str, default=None, help='Specify a cache dir.')
  parser.add_argument('-results','--results', type=str, default=None, help='Specify results dir.')

  args = parser.parse_args()

  return args

def load_and_build(model, args):
  # build the model 
  model.load_dataset()

  model.build_model()
  model.compile_model()

  if args.load_weights:
    model.load_weights(args.weight_name, absolute=True)


def exp(model, args):
  load_and_build(model, args)

  mask     = [ True for layer in model.get_layers()]

  #retrain  = dl_models.transform.Retraining(layer_mask=mask)
  sparsity = dl_models.transform.SummarizeSparsity(mode='both')
  distrib  = dl_models.transform.SummarizeDistribution()

  #retrain.config(model)

  print("=====================================================================")
  print('(0) Model Topology')
  print()
  for layer in model.get_layers():
    print('  ->',layer[0],':',layer[1].size())

  print('(1) Base model')
  print()
  err = model.eval_model()
  print('(1) error:',err)
  print()
  sparsity(model)
  print('(1) sparsity:',sparsity.get_summary())
  print()
  distrib(model)
  print('(1) distribution:',distrib.get_summary())
  print()

  layer_mask = [ True for layer in model.get_layers()]

  random_fault_injector = dl_models.transform.RandomFault(layer_mask, seed=args.seed,
                                                          frac=args.frate,
                                                          random_addrs=True,
                                                          fault_type='bit',
                                                          int_bits=args.qi,
                                                          frac_bits=args.qf)

  random_fault_injector(model)

  err = model.eval_model()
  print('(2) error after bit injection:',err)

  dargs = vars(args)
  for key in list(dargs.keys()):
    print("::::", key, dargs[key])
  print("::::", "error", err)

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

if __name__=='__main__':
  args = cli()
  np.random.seed(args.seed)
  config_setup(args)
  model_name = args.model
  ModelClass = model_class_map[model_name]
  model = ModelClass()
  print('Experimenting with model: %s' % model.model_name)
  exp(model, args)
