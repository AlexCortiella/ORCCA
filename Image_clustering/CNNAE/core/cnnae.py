## PyTorch
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
import torch.utils.data as data
import torch.optim as optim
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch.nn as nn

import torch
import torchvision.models as models

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim=2, fc2_input_dim=128, num_channels=[8, 16, 32], in_chan = 3):
        super().__init__()
        self.encoded_space_dim = encoded_space_dim
        self.num_channels = num_channels

        ### Convolutional section
        self.compress_layer = nn.Conv2d(3, 3, 3, stride=2, padding=0)

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, self.num_channels[0], 3, stride=2, padding=0),
            nn.BatchNorm2d(self.num_channels[0]),
            nn.ReLU(),
            nn.Conv2d(self.num_channels[0], self.num_channels[1], 3, stride=2, padding=0),
            nn.BatchNorm2d(self.num_channels[1]),
            nn.ReLU(),
            nn.Conv2d(self.num_channels[1], self.num_channels[2], 3, stride=2, padding=0),
            nn.ReLU()
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(9 * 9 * self.num_channels[2], fc2_input_dim),
            nn.ReLU(),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )

    def forward(self, x):
        for _ in range(3):
            x = self.compress_layer(x)
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim=2, fc2_input_dim=128, num_channels=[32, 16, 8], in_chan = 3):
        super().__init__()
        
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, fc2_input_dim),
            nn.ReLU(),
            nn.Linear(fc2_input_dim, 9 * 9 * num_channels[0]),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(num_channels[0], 9, 9))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(num_channels[0], num_channels[1], 3,
                               stride=2, output_padding=(0, 0)),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels[1], num_channels[2], 3, stride=2,
                               padding=0, output_padding=(0, 0)),
            nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels[2], in_chan, 3, stride=2,
                               padding=0, output_padding=(0, 0))
        )


        self.decompress_layer_0 = nn.ConvTranspose2d(in_chan, in_chan, 3, stride=2, padding=0, output_padding=(0,0))
        self.decompress_layer_1 = nn.ConvTranspose2d(in_chan, in_chan, 3, stride=2, padding=0, output_padding=(1,1))

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
#         x = self.decompress_layer_0(x)
        for _ in range(2):
            x = self.decompress_layer_0(x)
        x = self.decompress_layer_1(x)
        
        return x

class CNNAE(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        
        self.base_channel_size = 32
        self.latent_dim = cfg.latent_dim
        self.num_input_channels = 3
        self.width = 640
        self.height = 640
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = Encoder(encoded_space_dim = self.latent_dim)
        self.decoder = Decoder(encoded_space_dim = self.latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(8, self.num_input_channels, self.width, self.height)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)
    
    
