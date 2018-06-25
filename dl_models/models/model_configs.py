#BASE_DATASET_DIR = '/home/reagen/data/'
#BASE_CACHE_DIR = '/home/reagen/dl_models/cache/'

mnist_lenet5_params = { 'name': 'mnist_lenet5', 'learning_rate': 0.1, 'n_epochs': 200,
                      'dataset_name': 'mnist.pkl.gz', 'nkerns': [20, 50], 'batch_size': 128}

svhn_lenet5_params = { 'name': 'svhn_lenet5', 'learning_rate': 0.1, 'n_epochs': 200,
                      'dataset_name': 'svhn', 'nkerns': [20, 50], 'batch_size': 128,
                      'preprocessing_dir': "/data/svhn/preprocessed/"}

mnist_fc_params = { 'name': 'mnist_fc', 'learning_rate': 0.1, 'n_epochs': 200,
                      'dataset_name': 'mnist.pkl.gz', 'batch_size': 128}

tidigits_gru_params = { 'name': 'tidigits_gru', 'learning_rate': 1, 'n_epochs': 1,
                      'dataset_name': 'tidigits', 'batch_size': 32}

imagenetVGG16_params = { 'name': 'imagenetVGG16', 'learning_rate': 0.0001, 'n_epochs': 1,
     'dataset_name': 'imagenet', 'batch_size': 16,
     'preprocessing_dir': "/data/ilsvrc2012/preprocessed/"}

imagenetResNet50_params = { 'name': 'imagenetResnet50', 'learning_rate': 0.0001, 'n_epochs': 1,
     'dataset_name': 'imagenet', 'batch_size': 16,
     'preprocessing_dir': "/data/ilsvrc2012/preprocessed/"}

cifar10_vgg_params = { 'name': 'cifar10_vgg', 'learning_rate': 0.0001, 'n_epochs': 200,
                      'dataset_name': 'mnist.pkl.gz', 'nkerns': [20, 50], 'batch_size': 128}

model_params = {'mnist_lenet5'     : mnist_lenet5_params,
                'svhn_lenet5'      : svhn_lenet5_params,
                'mnist_fc'         : mnist_fc_params,
                'cifar10_vgg'      : cifar10_vgg_params,
                'imagenetVGG16'    : imagenetVGG16_params,
                'imagenetResNet50' : imagenetResNet50_params,
                'tidigits_gru'     : tidigits_gru_params}

# set the power of 2 rounding mode..
ROUND_FLOOR = True
