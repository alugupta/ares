import os.path
import numpy as np

from dl_models.configuration import Conf


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ModelBase(object):
  def __init__(self, dataset='mnist', model_name='abstract_model'):
    super(ModelBase, self).__init__()

    self.dataset = dataset
    self.model_name = model_name
    self.cache_dir = Conf.get('cache_dir')
    cache_file = '%s_%s' % (dataset, model_name)
    self.weights_file_name = os.path.join(self.cache_dir, cache_file)

    self.model = None

    self.traindata = None
    self.testdata = None
    self.valdata = None

    self.num_epochs = 10
    self.l1 = 0.0
    self.l2 = 0.0
    self.dropout_rate = 0.0

    self.layer_ids = []
    self.layer_prune_rates = []
    
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.total_weights = 0

  def set_model(self, model, layer_ids, layer_prune_rates):
    self.model = model
    self.layer_ids = layer_ids
    self.layer_prune_rates = layer_prune_rates
    self.total_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)

  def set_device(self, device):
    self.device = device

  def get_device(self):
    return self.device

  def set_training_params(self, args):
    print(args)
    self.num_epochs = args.epochs
    self.l2 = args.l2
    self.dropout_rate = args.dropout_rate

  def set_data(self, train, test, val):
    self.traindata = train
    self.testdata = test
    self.valdata = val

  def compile_model(self, loss='categorical_crossentropy', optimizer='adadelta', metrics=None):
        if loss == 'categorical_crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss == 'binary_crossentropy':
            self.criterion = nn.BCELoss()

        if optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), weight_decay=self.l2)
        elif optimizer == 'rms':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr = self.lr, weight_decay=self.l2)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr, weight_decay = self.l2, momentum=0.9, nesterov=True)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.l2)
        if metrics is None:
            self.metrics=['accuracy']
        else:
            self.metrics = metrics

  def save_best(self, best):
    new_err = test_model()
    if new_err < best:
        print("New best model with error %f", err)
        self.save_weights()
        return new_err
    return best

  def fit_model(self, batch_size=128, v=0, keep_best=False):
    best_acc = 1
    
    self.model.train()
    trainloader = torch.utils.data.DataLoader(self.traindata, batch_size=batch_size, shuffle=True, num_workers=1)
    self.model.to(self.device)
    for e in range(self.num_epochs):
        print("Training epoch " + str(e))
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        if keep_best:
            best_acc = self.save_best(best_acc)
            
  def accuracy(self, out, labels):
      vals, outputs = torch.max(out, dim=1)
      return torch.sum(outputs == labels).item()

  def check_model(self, dataset): 
    self.model.eval()
    losses = []
    count = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=512,shuffle=False,num_workers=1)
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            if 'accuracy' in self.metrics:
                loss = self.accuracy(self.model(inputs).data,labels.data)
                losses.append(loss)
                count += list(labels.data.size())[0]
    return 1. - np.sum(losses) / count


  def eval_model(self, v=0):
    return self.check_model(self.valdata)
  def test_model(self, v=0):
    return self.check_model(self.testdata)

  #Foward hooks, used to handle activation injections
  def register_hook(self, hook, module_ind):
    list(self.model.modules())[module_ind].register_forward_hook(hook)
  def get_modules(self):
    return self.model.modules()

  def save_weights(self, filename=None):
    if filename:
      fname = filename
    else:
      fname = self.weights_file_name

    print("FILENAME = ", fname)
    torch.save(self.model.state_dict(), fname)

  def load_weights(self, fname=None, absolute=False):
    print("Loading weights at:",fname)
    self.model.load_state_dict(torch.load(fname))
    self.model.to(self.device)

  #Get layers _excluding_ bias layers, (use include_biases flag in transforms to control them)
  def get_layers(self):
    layers = list(self.model.named_parameters())
    nb_layers = []
    for layer in layers:
        if 'bias' not in layer[0]:
            nb_layers.append(layer)
    return nb_layers

  #Internal method used to get all layers, including biases
  def get_all_layers(self): 
    return list(self.model.named_parameters())

  def update_layer(self, layer, new_data):
    layer[1].data = new_data.to(self.device)

class IndirectModel(ModelBase):
  '''A model with indirect indices instead of a normal weight matrix.

  This model cannot be executed directly, as its weights are effectively
  meaningless.
  '''

  def __init__(self, *args, **kwargs):
    super(IndirectModel,self).__init__(*args, **kwargs)
    self.value_table = None

  _NIE_msg = 'Cannot execute this method on an indirect model. Convert to a regular model first.'
  def compile_model(self, *a, **k): raise NotImplementedError(self._NIE_msg)
  def fit_model(self, *a, **k): raise NotImplementedError(self._NIE_msg)
  def eval_model(self, *a, **k): raise NotImplementedError(self._NIE_msg)
  def test_model(self, *a, **k): raise NotImplementedError(self._NIE_msg)
  def save_weights(self, *a, **k): raise NotImplementedError(self._NIE_msg)
  def load_weights(self, *a, **k): raise NotImplementedError(self._NIE_msg)

  def get_values(self):
    '''Return the actual weight values shared by the model.'''
    return self.value_table
