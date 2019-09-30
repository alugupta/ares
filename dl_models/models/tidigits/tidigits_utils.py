import numpy as np
import pickle, hickle
import sys
import glob
import json
import scipy
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

import h5py


tidigits_dataset_path = '/group/vlsiarch/lpentecost/dl-models/dl_models/models/tidigits/tidigits.hdf5'

class tidigits(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).type(torch.long)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
  # Converts single element of tidigits targets (y_train and y_test) from type
  # string to type int
def convert_tidigits(x):
    if x == b'Z' or x == b'O':
      return 0
    else:
      return int(x)

    # Read in tidigits.hdf5. Assumes dataset is one directory up
def read_tidigits(fname):
    tidigits = h5py.File(fname, 'r')
    print (tidigits)

    print((list(tidigits.keys())))

    for i, key in enumerate(tidigits.keys()):
      if i == 0:
        x_train_key = key
      elif i == 1:
        x_test_key =  key
      elif i == 2:
        y_train_key =  key
      elif i == 3:
        y_test_key =  key


    # Training Set
    x_train = tidigits[x_train_key]
    y_train = tidigits[y_train_key]

    # Testing Set
    x_test = tidigits[x_test_key]
    y_test = tidigits[y_test_key]

    # Original labels for test and training set are in string format. We must
    # convert them into integers as strings cannot be made into tensor types
    y_test  = [convert_tidigits(s) for s in y_test]
    y_train = [convert_tidigits(s) for s in y_train]

    data = (np.array(x_train).astype('float32'), np.array(y_train).astype('long'),  np.array(x_test).astype('float32') , np.array(y_test).astype('long'))
    return data

def preprocess_categorical(ys, yclasses=2, dtype='float32'):
    new_ys = []
    for y in ys:
      # Convert categorical labels to binary (1-hot) encoding
      y = np.eye(yclasses)[y]
      new_ys.append(y)
    return np.array(new_ys)

def load_dataset():
    return read_tidigits(tidigits_dataset_path)



    
