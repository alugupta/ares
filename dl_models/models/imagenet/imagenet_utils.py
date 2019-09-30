import numpy as np
import sys
import glob
import json
import scipy
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

class imageNet(Dataset):        #Dataset to handle train and test sets
    def __init__(self, img_dir, labels, transform=None, device='cpu'):
        self.img_dir = img_dir
        self.labels = torch.tensor(labels, dtype=torch.long, device=device) 
        self.batch_size = 256
        self.img_pickles = sorted(glob.glob(self.img_dir + "*.pt"))
        self.device = device

    def __len__(self):
        return len(self.img_pickles)
    
    def __getitem__(self, i):
        X = self.load_pickled_imgs(i)
        Y = self.labels [self.batch_size * i : self.batch_size * (i + 1) ]

        # assert that the number of images in X_train and Y_train is the same
        assert (X.shape[0] == Y.shape[0])
        return X, Y

    def load_pickled_imgs(self, batchid):
        path =  self.img_pickles[batchid] 
        return torch.load(path, map_location = self.device)

''' Loads train.npy, val.npy and shuffled_train_filenames.npy'''
def load_imagenet_labels(train_label_filename, val_label_filename):
    train_labels             = np.load(train_label_filename)
    val_labels               = np.load(val_label_filename)

    return (train_labels, val_labels)
