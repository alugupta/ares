import dl_models
import argparse

from keras.layers import Dense
from keras.models import load_model
from keras import backend as K

from dl_models.models.model_configs import *
from dl_models.models.base import *

from dl_models.models import mnistFC
from dl_models.models import mnistLenet5
from dl_models.models import svhnLenet5
from dl_models.models import imagenetVGG16
from dl_models.models import imagenetResNet50
from dl_models.models import cifar10VGG

model_class_map = {
                   'mnist_lenet5'      : mnistLenet5,
                   'mnist_fc'          : mnistFC,
                   'svhn_lenet5'       : svhnLenet5,
                   'imagenet_vgg16'    : imagenetVGG16,
                   'imagenet_resnet50' : imagenetResNet50,
                   'cifar10_vgg'   : cifar10VGG,
                  }

def cli():
  # Default message.
  parser = argparse.ArgumentParser(description='Quantization experiment', \
                                        epilog='Configure you experiment from the command line.')

  parser.add_argument('-m', '--model', required=True, type=str, \
                       help='Pick a model to run. Models listed in models/model_config.py')

  parser.add_argument('-lw', '--load_weights', action='store_true', help='Load saved weights from cache.')
  parser.add_argument('-sw', '--save_path', default=None, type=str, help='Path to save weights')

  parser.add_argument('-ld_name', '--weight_name', default=None, type=str, \
                           help='Specifiy the weights to use.')

  parser.add_argument('-qi', '--qi', default=2, type=int, help='Integer bits for quantization')
  parser.add_argument('-qf', '--qf', default=6, type=int, help='Fractional bits for quantization')

  parser.add_argument('-c','--configuration', type=str, default=None, help='Specify a configuration file.')

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

  model.model.summary()

def quantize_exp(model, mask, args):
  q = (args.qi, args.qf)
  print "====================================================================="
  print "Quantize experiment args: "
  print "  mask: ", mask
  print "  q: ", q

  errs       = [ ]

  print model.eval_model()
  quantizer = dl_models.transform.Quantize(mask, q)
  quantizer(model)
  err = model.eval_model() #eval after quantizing

  print "Quantization: ", q, " \t | Error = ", err

  model.save_weights(args.save_path + '_quantized_' \
                    + str(q[0]) + "_" + str(q[1]))

def exp(model, args):
  load_and_build(model, args)

  mask     = [ True for layer in model.get_layers()]

  print "====================================================================="
  print '(0) Model Topology'
  print
  for layer in model.get_layers():
    print '  ->',layer,':',','.join([str(w.shape) for w in layer.get_weights()])

  quantize_exp(model, mask, args)

def config_setup(args):
  if args.configuration is not None:
    Conf.load(Conf.find_config(args.configuration))
  else:
    Conf.set_env_default()

  if args.cache is not None:
    Conf.set('cache', args.cache)

  if args.results is not None:
    Conf.set('results', args.results)

if __name__=='__main__':
  args = cli()
  config_setup(args)
  model_name = args.model
  ModelClass = model_class_map[model_name]
  model = ModelClass()
  print 'Experimenting with model: %s' % model.model_name
  exp(model, args)
