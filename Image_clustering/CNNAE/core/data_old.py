# DATA

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

#import torchvision
#import torchvision.transforms as transforms
#from torchvision.io import read_image

import pandas as pd
import numpy as np
import os
import json


from torch.utils.data.sampler import SubsetRandomSampler

#### DATA LOADER ####

class SurfReacDataModule(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.n_train = cfg.n_train
        self.n_val = cfg.n_val
        self.n_test = cfg.n_test
        self.n_samples = self.n_train + self.n_val + self.n_test
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.cuda = cfg.cuda


    def setup(self, stage = None):
        
        
        f = open(self.data_dir)
        data_dict = json.load(f)
        
        assert(self.n_samples <= len(data_dict))

        data_tr = []
        label_tr = []
        data_val = []
        label_val = []
        data_test = []
        label_test = []
        
        for i in range(self.n_samples):

            if i>=0 and i < self.n_train:
                data_tr.append(torch.tensor(data_dict[f'sample{i+1}']['data'], dtype = torch.float))
                label_tr.append(torch.tensor(data_dict[f'sample{i+1}']['label'], dtype = torch.uint8))
            elif i>=self.n_train and i < self.n_train + self.n_val:
                data_val.append(torch.tensor(data_dict[f'sample{i+1}']['data'], dtype = torch.float))
                label_val.append(torch.tensor(data_dict[f'sample{i+1}']['label'], dtype = torch.uint8))
            else:
                data_test.append(torch.tensor(data_dict[f'sample{i+1}']['data'], dtype = torch.float))
                label_test.append(torch.tensor(data_dict[f'sample{i+1}']['label'], dtype = torch.uint8))        

        self.train_dataset = tuple(zip(data_tr, label_tr))
        self.val_dataset = tuple(zip(data_val, label_val))
        self.test_dataset = tuple(zip(data_test, label_test))
        
        self.input_dim = len(self.train_dataset[0][0])

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_stage_dataset =  self.train_dataset
            self.val_stage_dataset = self.val_dataset

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_stage_dataset = self.test_dataset

    def train_dataloader(self):
        if self.cuda:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False, num_workers = self.num_workers, sampler=DistributedSampler(self.train_dataset))
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)

    def val_dataloader(self):
        if self.cuda:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False, num_workers = self.num_workers, sampler=DistributedSampler(self.val_dataset))
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)

    def test_dataloader(self):
        if self.cuda:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False, num_workers = self.num_workers, sampler=DistributedSampler(self.test_dataset))
        else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)

