#MODEL

###### MODEL CLASS ######

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
from  torch.distributions.categorical import Categorical

import pandas as pd
import numpy as np


#Autoencoder definition
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
            nn.Linear(fc2_input_dim, encoded_space_dim * 2)
        )

    def forward(self, x):
        for _ in range(3):
            x = self.compress_layer(x)
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        mu, logsigmasq = x[:, :self.encoded_space_dim], x[:, self.encoded_space_dim:]
        return mu, logsigmasq


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim=2, fc2_input_dim=128, num_channels=[32, 16, 8], in_chan = 3, output_var = None):
        super().__init__()
        
        if not(isinstance(output_var, int) or isinstance(output_var, float)):
            self.var_x = None
        else:
            self.var_x = torch.tensor(output_var)

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
            nn.ConvTranspose2d(num_channels[2], 2*in_chan, 3, stride=2,
                               padding=0, output_padding=(0, 0))
        )


        self.decompress_layer_0 = nn.ConvTranspose2d(2*in_chan, 2*in_chan, 3, stride=2, padding=0, output_padding=(0,0))
        self.decompress_layer_1 = nn.ConvTranspose2d(2*in_chan, 2*in_chan, 3, stride=2, padding=0, output_padding=(1,1))

    def forward(self, x):

        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
#         x = self.decompress_layer_0(x)
        for _ in range(2):
            x = self.decompress_layer_0(x)
        x = self.decompress_layer_1(x)
        
        if self.var_x is None:
            mu, logsigmasq = x[:, :3, :, :], x[:, 3:, :, :]
        else:
            mu = x[:, :3, :, :]
            logsigmasq = torch.ones_like(mu) * torch.log(self.var_x)
            
        return mu, logsigmasq

#GMVAE intermediate functions

class GMVAE(nn.Module):
    
    
    def __init__(self, cfg):
        super().__init__()
        # initialize latent GMM model parameters
        
        self.latent_dim = cfg.latent_dim
        self.n_clusters = cfg.n_clusters
        self.n_modalities = cfg.n_modalities
        self.num_epochs = cfg.total_epochs
        self.seed = cfg.seed
        self.w_rec = cfg.w_rec
        self.w_reg = cfg.w_reg
        self.w_entr = cfg.w_entr
        
        if isinstance(cfg.output_variance, int) or isinstance(cfg.output_variance, float):
            self.logsigmasq_x = cfg.output_variance
        else:
            self.logsigmasq_x = None
            
        
        self.params = {}
        self.pi_variables = nn.Parameter(torch.zeros(self.n_clusters), requires_grad = True)
        self.params['pi_c'] = torch.ones(self.n_clusters) / self.n_clusters
        
        torch.manual_seed(self.seed)
        self.mu_c = nn.Parameter(torch.rand((self.n_clusters, self.latent_dim))*2.0 - 1.0, requires_grad = False)
        self.logsigmasq_c = nn.Parameter(torch.zeros((self.n_clusters, self.latent_dim)), requires_grad = False)
        self.params['mu'] = None
        self.params['logsigmasq'] = None
        self.params['z'] = None
        self.params['predicted_clusters'] = None

        
        self.params['hist_weights'] = torch.zeros((self.n_clusters, 1)).clone().detach()
        self.params['hist_mu_c'] = torch.zeros((self.n_clusters, self.latent_dim)).clone().detach()
        self.params['hist_logsigmasq_c'] = torch.zeros((self.n_clusters, self.latent_dim)).clone().detach()
        
        # initialize neural networks
        torch.manual_seed(self.seed)
        self.encoder = Encoder(encoded_space_dim=self.latent_dim)

        torch.manual_seed(self.seed)
        self.decoder = Decoder(encoded_space_dim=self.latent_dim, output_var = self.logsigmasq_x)
        #self.encoder_parameters = nn.Parameter(encoder.parameters())
        #self.decoder_parameters = nn.Parameter(decoder.parameters())
            
        # Utils
        self.em_reg = cfg.numerical_tolerance
        
    def _encoder_step(self, x_list, encoder, decoder):
        """
        Maps D-modality data to distributions of latent embeddings.
        :param x_list: length-D list of (N, data_dim) torch.tensor
        :param encoder_list: length-D list of Encoder
        :param decoder_list: length-D list of Decoder
        :param params: dictionary of non-DNN parameters
        :return:
            mu: (N, latent_dim) torch.tensor containing the mean of embeddings
            sigma: (N, latent_dim) torch.tensor containing the std dev of embeddings
        """
        mu, logsigmasq = encoder.forward(x_list[0])

        return mu, logsigmasq + self.em_reg
    
    def _em_step(self, z, mu, update_by_batch=False):
        # compute gamma_c ~ p(c|z) for each x
        mu_c = self.mu_c  # (K, Z)
        logsigmasq_c = self.logsigmasq_c  # (K, Z)
        sigma_c = torch.exp(0.5 * logsigmasq_c)
        pi_c = self.params['pi_c']

        log_prob_zc = Normal(mu_c, sigma_c).log_prob(z.unsqueeze(dim=1)).sum(dim=2) + torch.log(pi_c)  #[N, K]
        log_prob_zc -= log_prob_zc.logsumexp(dim=1, keepdims=True)

        gamma_c = torch.exp(log_prob_zc) + self.em_reg
        
        denominator = torch.sum(gamma_c, dim=0).unsqueeze(1)
        mu_c = torch.einsum('nc,nz->cz', gamma_c, mu) / denominator
        logsigmasq_c = torch.log(torch.einsum('nc,ncz->cz', gamma_c, (mu.unsqueeze(dim=1) - mu_c) ** 2)) - torch.log(denominator)
        
        if not update_by_batch:
            return gamma_c, mu_c, logsigmasq_c

        else:
            hist_weights = self.params['hist_weights'].to(self.gpu_id)
            hist_mu_c = self.params['hist_mu_c'].to(self.gpu_id)
            hist_logsigmasq_c = self.params['hist_logsigmasq_c'].to(self.gpu_id)

            curr_weights = denominator
            new_weights = hist_weights + curr_weights
            new_mu_c = (hist_weights * hist_mu_c + curr_weights * mu_c) / new_weights
            new_logsigmasq_c = torch.log(hist_weights * torch.exp(hist_logsigmasq_c) + curr_weights * torch.exp(logsigmasq_c)) - torch.log(new_weights)
            #new_logsigmasq_c = torch.log(torch.exp(torch.log(hist_weights) + hist_logsigmasq_c) +
            #                             torch.exp(torch.log(curr_weights) + logsigmasq_c)) - torch.log(new_weights)

            self.params['hist_weights'] = new_weights
            self.params['hist_mu_c'] = new_mu_c
            self.params['hist_logsigmasq_c'] = new_logsigmasq_c
            return gamma_c, new_mu_c, new_logsigmasq_c
        
        
    def _decoder_step(self, x_list, z, encoder, decoder, mu, logsigmasq, gamma_c):
        """
        Computes a stochastic estimate of the ELBO.
        :param x_list: length-D list of (N, data_dim) torch.tensor
        :param z: MC samples of the encoded distributions
        :param encoder_list: length-D list of Encoder
        :param decoder_list: length-D list of Decoder
        :param params: dictionary of non-DNN parameters
        :return:
        elbo: (,) tensor containing the elbo estimation
        """
        sigma = torch.exp(0.5 * logsigmasq)
        mu_c = self.mu_c
        logsigmasq_c = self.logsigmasq_c
        pi_c = self.params['pi_c']
        elbo = 0
    
        reconstruction = 0
        regularization = 0
        entropy = 0
        
        #Variable or fixed output variance

        mu_x, logsigmasq_x = decoder.forward(z)
        
        #torch.manual_seed(self.seed)
        reconstruction = Normal(mu_x, torch.exp(0.5 * logsigmasq_x)).log_prob(x_list[0]).sum()
        #mse_loss = nn.MSELoss()
        #reconstruction = -mse_loss(x_list[0], mu_x)
        
        Nb = x_list[0].shape[0]
        reconstruction = 1/Nb * reconstruction 
        
        regularization = - 1/(2*Nb)*torch.sum(gamma_c * (logsigmasq_c + (sigma.unsqueeze(1) ** 2 + (mu.unsqueeze(1) - mu_c) ** 2) /
                                         torch.exp(logsigmasq_c)).sum(dim=2)) + 1/(2*Nb) * torch.sum(1 + logsigmasq)
        
        entropy = 1/(2*Nb)*torch.sum(gamma_c * (torch.log(pi_c) - torch.log(gamma_c))) 

        elbo = self.w_rec*reconstruction + self.w_reg*regularization + self.w_entr*entropy / (self.w_rec + self.w_reg + self.w_entr)
        
        return elbo, reconstruction, regularization, entropy
        
    def loss(self, batch_x, mode = 'train'):
        
        self.gpu_id = batch_x.get_device()
           
        #Extract data modalities
        x_list = [batch_x]  # assume D=2 and each modality has data_dim
        
        #Assign pi_c
        if mode == 'train':
            pi_c = torch.exp(self.pi_variables) / torch.sum(torch.exp(self.pi_variables))
            self.params['pi_c'] = pi_c
            
        #Encode input data to Gaussian latent space with mu and logsigmaq
        mu_z, logsigmasq_z = self._encoder_step(x_list, self.encoder, self.decoder)

        #Sample from the latent space
        sigma_z = torch.exp(0.5 * logsigmasq_z)
        torch.manual_seed(self.seed)
        eps = Normal(0, 1).sample(mu_z.shape).to(self.gpu_id)
        z = mu_z + eps * sigma_z
        self.params['z'] = z

        #Perform EM step to estimate mu_c and logsigmasq_c
        with torch.no_grad():
            gamma_c, mu_c, logsigmasq_c = self._em_step(z, mu_z, update_by_batch=True)
            
        self.params['predicted_clusters'] = torch.argmax(gamma_c, axis = 1)
            
        if mode == 'train':
            self.mu_c.data = mu_c
            self.logsigmasq_c.data = logsigmasq_c
        
        #Compute elbo and loss
        elbo, reconstruction, regularization, entropy = self._decoder_step(x_list, z, self.encoder, self.decoder, mu_z, logsigmasq_z, gamma_c)
        
        loss = - elbo
        rec_loss = - reconstruction
        reg_loss = - regularization
        entropy_loss = - entropy

        return loss, rec_loss, reg_loss, entropy_loss
        
    def generate(self, n_samples):
        
        pi_c =  torch.exp(self.pi_variables.detach()) / torch.sum(torch.exp(self.pi_variables.detach()))
        pi_c = pi_c / torch.sum(pi_c)#Make sure probabilities add to 1
        
        #Extract cluster means and variances
        mu_c = self.mu_c.detach()
        sigma_c = torch.exp(0.5 * self.logsigmasq_c.detach())
        
        X = np.zeros((self.input_dim, n_samples))
        Z = np.zeros((self.latent_dim, n_samples))
        C = np.zeros((n_samples, ))
        
        #Generate samples
        for s in range(n_samples):
        
            #Sample cluster
            c = np.random.choice(self.n_clusters, size = 1, p = pi_c.numpy())
            
            #Extract latent mean and variance
            mu_z, sigma_z = mu_c[c,:], sigma_c[c,:]
            
            #Generate sample in latent space
            z = mu_z + torch.randn(1,mu_z.shape[1]) * sigma_z

            #Decode sample in latent space into input space mean and variance

            mu_x, logsigmasq_x = self.decoder.forward(z)
                
            sigma_x = torch.exp(0.5*logsigmasq_x)
            
            #Generate sample in input space    
            x = mu_x + 0*torch.randn(1,mu_x.shape[1]) * sigma_x
            
            #Store sample
            X[:,s] = x.detach().numpy()
            Z[:,s] = z.detach().numpy()
            C[s] = c
            
        return C, Z, X
            
            
    def predict(self, batch_x):
        
        with torch.no_grad():
            #Extract data modalities
            x_list = [batch_x]  # assume D=2 and each modality has data_dim

            #Assign pi_c
            pi_c =  torch.exp(self.pi_variables.detach()) / torch.sum(torch.exp(self.pi_variables.detach()))
            pi_c = pi_c / torch.sum(pi_c)#Make sure probabilities add to 1

            #Encode input data to Gaussian latent space with mu and logsigmaq
            mu_z, logsigmasq_z = self._encoder_step(x_list, self.encoder_list, self.decoder_list)

            #Sample from the latent space
            #torch.manual_seed(self.seed)
            sigma_z = torch.exp(0.5 * logsigmasq_z)
            eps = Normal(0, 1).sample(mu_z.shape)
            z = mu_z + eps * sigma_z

            # compute gamma_c ~ p(c|z) for each x
            mu_c = self.params['mu_c']  # (K, Z)
            logsigmasq_c =self.params['logsigmasq_c']  # (K, Z)
            sigma_c = torch.exp(0.5 * logsigmasq_c)

            log_prob_zc = Normal(mu_c, sigma_c).log_prob(z.unsqueeze(dim=1)).sum(dim=2) + torch.log(pi_c)  #[N, K]
            log_prob_zc -= log_prob_zc.logsumexp(dim=1, keepdims=True)
            gamma_c = torch.exp(log_prob_zc) + self.em_reg
            

            mu_x, logsigmasq_x = self.decoder.forward(z)
                
            #Sample from the input space
            #torch.manual_seed(self.seed)
            sigma_x = torch.exp(0.5 * logsigmasq_x)
            eps_x = Normal(0, 1).sample(mu_x.shape)
            x_rec = mu_x + eps_ * sigma_x
            
        return mu_z, logsigmasq_z, z, gamma_c, mu_, logsigmasq_, x_rec
