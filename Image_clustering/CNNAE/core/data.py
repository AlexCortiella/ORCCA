# DATA

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
import matplotlib.image as mpimg

import pandas as pd
import numpy as np
import os
import json


from torch.utils.data.sampler import SubsetRandomSampler

## Data loader

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        img_dir = os.path.join(data_dir,'images')
        lbl_dir = os.path.join(data_dir,'labels')
        self.img_ID = [os.path.splitext(img)[0] for img in os.listdir(img_dir)]
        self.img_type = [os.path.splitext(img)[1] for img in os.listdir(img_dir)]
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transform = lambda x : -1 + 2*(x/torch.max(x))
        #self.transform = torchvision.transforms.Normalize()
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_ID)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.img_ID[idx]}{self.img_type[idx]}')
        image = read_image(img_path, torchvision.io.ImageReadMode.UNCHANGED).to(torch.float32)
        
        #lbl_path = open(os.path.join(self.lbl_dir, f'{self.img_ID[idx]}.txt'),'r')
        #lines = lbl_path.readlines()
        
#         labels = torch.tensor([lbl[0] for lbl in lines])
        labels = 0

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        return image, labels
    
class ImageDataModule():
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.batch_size = cfg.batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.target_transform = transforms.Compose([transforms.ToTensor()])
        self.cuda = cfg.cuda

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.image_train = ImageDataset(self.data_dir + '/train/')
            self.image_val = ImageDataset(self.data_dir + '/valid/')

        if stage == "test":
            self.image_test = ImageDataset(self.data_dir + '/test/')
    def train_dataloader(self):
        
        if self.cuda:
            sampler = DistributedSampler(self.image_train)
        else:
            sampler = None
            
        return DataLoader(self.image_train,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          shuffle=False,
                          sampler = sampler)
         
    def valid_dataloader(self):
        
        
        if self.cuda:
            sampler = DistributedSampler(self.image_val)
        else:
            sampler = None
            
        return DataLoader(self.image_val,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          shuffle=False,
                          sampler=sampler)
    

    def test_dataloader(self):
        return DataLoader(self.image_test, batch_size=self.batch_size)
        
        
## Data loader

def read_labels(labels_path):
    d = {}
    f = open(labels_path, "r")
    for line in f:
        (key, val) = line.split()
        d[key] = int(val)
    return d

class KaggleDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        img_dir = os.path.join(data_dir,'images')
        lbl_dir = os.path.join(data_dir,'labels')
        
        self.img_ID = [os.path.splitext(img)[0] for img in os.listdir(img_dir)]
        self.img_type = [os.path.splitext(img)[1] for img in os.listdir(img_dir)]
        self.labels = read_labels(os.path.join(lbl_dir,'labels.txt'))
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.reescale = lambda x : -1 + 2*(x/torch.max(x))
        self.transform = transforms.Compose([transform, self.reescale])
        self.target_transform = target_transform
        self.current_ID = None

    def __len__(self):
        return len(self.img_ID)

    def __getitem__(self, idx):
        
        ID = self.img_ID[idx]
        typ = self.img_type[idx]
        self.current_ID = f'{ID}{typ}'
        label = torch.tensor(int(self.labels[ID+typ]), dtype = torch.uint8)
        img_path = os.path.join(self.img_dir, f'{ID}{typ}')

        image = mpimg.imread(img_path)
        if image.ndim < 3: #Grayscale image
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        else:
            image = image[:,:,0:3]
            
        image = torch.permute(torch.tensor(image, dtype = torch.float), (2,0,1))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return (image, label)
    
class KaggleDataModule():
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.batch_size = cfg.batch_size
        self.img_size = (int(cfg.img_size[0]), int(cfg.img_size[1]))
        self.n_train = cfg.n_train
        self.n_valid = cfg.n_valid
        self.n_test = cfg.n_test
        self.n_total = self.n_train + self.n_valid + self.n_test
        self.transform = transforms.Compose([transforms.CenterCrop(self.img_size)])
#         self.target_transform = transforms.Compose([transforms.ToTensor()])
        self.cuda = cfg.cuda
        self.dataset = KaggleDataset(self.data_dir, transform=self.transform)
        self.idxs = np.random.randint(0, high=len(self.dataset)-1, size=self.n_total)

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        
        if stage == "fit":
            
            self.image_train = []
            self.image_valid = []
            
            for i in range(self.n_train):
            
                try:
                    self.image_train.append(self.dataset[self.idxs[i]])
                except:
                    print(f"File corrupted - {self.dataset.current_ID} - Train. Ignoring file...")
                    continue
            for i in range(self.n_train,self.n_train + self.n_valid):
                try:
                    self.image_valid.append(self.dataset[self.idxs[i]])
                except:
                    print(f"File corrupted - {self.dataset.current_ID} - Valid. Ignoring file...")
                    continue

        if stage == "test":
            self.image_test = []
            for i in range(self.n_train + self.n_valid,self.n_total):
                try:
                    self.image_test.append(self.dataset[self.idxs[i]])
                except:
                    print(f"File corrupted - {self.dataset.current_ID} - Test. Ignoring file...")
                    continue
        if stage == "outliers":
            self.image_outliers = ImageDataset(self.data_dir + '/outliers/', transform=self.transform)
    def train_dataloader(self):
            
        return DataLoader(self.image_train,
                          batch_size=self.batch_size)
         
    def valid_dataloader(self):
            
        return DataLoader(self.image_valid,
                          batch_size=self.batch_size)
    

    def test_dataloader(self):
        return DataLoader(self.image_test, batch_size=self.batch_size)
    
    def outliers_dataloader(self):
        return DataLoader(self.image_outliers, batch_size=self.batch_size)
