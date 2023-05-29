####
## Standard libraries
import os
import json
import math
import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
#import seaborn as sns

## Progress bar
from tqdm.notebook import tqdm

# Pytorch
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

# Torchvision
import torchvision
from torchvision import transforms

# Pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cnnae import CNNAE
from data import KaggleDataset, KaggleDataModule


############ FUNCTION DEFINITIONS ###################
def get_train_images(num):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)

    
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
            
            
def train_model(latent_dim):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"cats_and_dogs_{latent_dim}"),
                         accelerator="gpu",
                         devices=2,
                         max_epochs=200,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    GenerateCallback(get_train_images(8), every_n_epochs=10),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"cats_and_dogs_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = CNNAE.load_from_checkpoint(pretrained_filename)
    else:
        model = CNNAE(base_channel_size=32, latent_dim=latent_dim)
        trainer.fit(model, train_loader, valid_loader)
    # Test best model on validation and test set
    valid_result = trainer.test(model, valid_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": valid_result}
    return model, result


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




#######################################################################################################



if __name__ == "__main__":


	# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
	DATASET_PATH = './data/kaggle_cats_and_dogs/PetImages'
	# Path to the folder where the pretrained models are saved
	CHECKPOINT_PATH = "./results/saved_models/new_results"

	# Setting the seed
	pl.seed_everything(42)

	# Ensure that all operations are deterministic on GPU (if used) for reproducibility
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	print("Device:", device)


	### DATALOADER ###

	data_dir = './data/kaggle_cats_and_dogs/PetImages'
	datamodule = KaggleDataModule(data_dir, (640,640))

	datamodule.setup('fit')
	datamodule.setup('test')

	train_loader = datamodule.train_dataloader()
	valid_loader = datamodule.valid_dataloader()
	test_loader = datamodule.test_dataloader()

	train_dataset = train_loader.dataset



	model_dict = {}
	for latent_dim in [2, 8, 16, 32, 64]:
	    model_ld, result_ld = train_model(latent_dim)
	    model_dict[latent_dim] = {"model": model_ld, "result": result_ld}
	    
	latent_dims = sorted([k for k in model_dict])
	val_scores = [model_dict[k]["result"]["val"][0]["test_loss"] for k in latent_dims]

	fig = plt.figure(figsize=(6,4))
	plt.plot(latent_dims, val_scores, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
	plt.xscale("log")
	plt.xticks(latent_dims, labels=latent_dims)
	plt.title("Reconstruction error over latent dimensionality", fontsize=14)
	plt.xlabel("Latent dimensionality")
	plt.ylabel("Reconstruction error")
	plt.minorticks_off()
	plt.ylim(0,100)
	plt.show()
	plt.savefig("AE_latent_reconstrunction.png", dpi=600)

