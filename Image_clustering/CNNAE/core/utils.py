#UTILITIES
import yaml
import torch
from core.data import KaggleDataModule
from core.cnnae import CNNAE
from torch.distributed import init_process_group, destroy_process_group
import os
import pytorch_lightning as pl

#Configuration file reader utilities
class dic2struc():
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])
            
def struct2dic(struc):

    return struc.__dict__
    

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)

def load_train_objs(cfg):
    
    #Prepare data
    datamodule = KaggleDataModule(cfg)
    datamodule.setup('fit')# load your dataset
    datamodule.setup('test')# load your dataset
    
    train_set = datamodule.train_dataloader()#train DataLoader   
    valid_set = datamodule.valid_dataloader() #val DataLoader
    test_set = datamodule.test_dataloader() #val DataLoader
    
    print(len(train_set))

    #Instantiate model
    model = CNNAE(cfg)  # load your model
    
    #IConfigure optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    return train_set, valid_set, test_set, model, optimizer
    
    
def get_train_images(num):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)
    
    
def visualize_reconstructions(model, input_imgs):
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(input_imgs.to(model.device))
    reconst_imgs = reconst_imgs.cpu()
    
    # Plotting
    imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, range=(-1,1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7,4.5))
    plt.title(f"Reconstructed from {model.hparams.latent_dim} latents")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

    
class GenerateCallback(pl.Callback):

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)
    
### OPERATIONS AND METRICS ##########

# Compute MMD (maximum mean discrepancy) using numpy and scikit-learn.

import numpy as np
from sklearn import metrics

class MMD():

    def __init__(self, X, Y, kernel = 'RBF'):
    
        self.X = X
        self.Y = Y
        self.kernel = kernel
    
    def mmd_linear(X, Y):
        """MMD using linear kernel (i.e., k(x,y) = <x,y>)
        Note that this is not the original linear MMD, only the reformulated and faster version.
        The original version is:
            def mmd_linear(X, Y):
                XX = np.dot(X, X.T)
                YY = np.dot(Y, Y.T)
                XY = np.dot(X, Y.T)
                return XX.mean() + YY.mean() - 2 * XY.mean()
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Returns:
            [scalar] -- [MMD value]
        """
        delta = X.mean(0) - Y.mean(0)
        return delta.dot(delta.T)


    def mmd_rbf(X, Y, gamma=1.0):
        """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            gamma {float} -- [kernel parameter] (default: {1.0})
        Returns:
            [scalar] -- [MMD value]
        """
        XX = metrics.pairwise.rbf_kernel(X, X, gamma)
        YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
        XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()


    def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
        """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            degree {int} -- [degree] (default: {2})
            gamma {int} -- [gamma] (default: {1})
            coef0 {int} -- [constant item] (default: {0})
        Returns:
            [scalar] -- [MMD value]
        """
        XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
        YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
        XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
        return XX.mean() + YY.mean() - 2 * XY.mean()
        
    def compute(self):
        
        MMD = None
        if self.kernel == 'RBF':
            MMD = self.mmd_rbf(self.X, self.Y)
        elif self.kernel == 'polynomial':
            MMD = self.mmd_poly(self.X, self.Y)
        elif self.kernel == 'linear':
            MMD = self.mmd_linear(self.X, self.Y)
            
        else:
            print(f"kernel not implemented. Please select one option {['linear','RBF', 'polynomial']}")
            
        return MMD
