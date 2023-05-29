# TRAINER
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tensorboard_logger import configure, log_value
import os

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        total_epochs: int,
        total_episodes: int,
        logs_path: str,
        restart: bool = False,
        cuda: bool = False
    ) -> None:
        
        if cuda:
            self.gpu_id = int(os.environ["LOCAL_RANK"])
            self.model = model.to(self.gpu_id)
            self.model = DDP(self.model, device_ids=[self.gpu_id])
        else:
            self.model = model
        
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.save_flag = False
        self.total_epochs = total_epochs
        self.total_episodes = total_episodes
        self.epochs_run = 0
    
        self.logs_path = logs_path
        self.restart = restart
        self.run_path = None
        self.current_run = 0
        
        self.cuda = cuda        
        
        self.epoch_list = np.zeros(1)
        self.train_loss = np.zeros(1)
        self.val_loss = np.zeros(1)
        self.train_rec_loss = np.zeros(1)
        self.train_reg_loss = np.zeros(1)
        self.train_entr_loss = np.zeros(1)
        self.val_rec_loss = np.zeros(1)
        self.val_reg_loss = np.zeros(1)
        self.val_entr_loss = np.zeros(1)
        
        self.pi_history = np.expand_dims(np.zeros_like(self.model.params['pi_c']), axis = 1)

        self.mu_c_history = np.expand_dims(np.zeros_like(self.model.mu_c), axis = 2)
        self.logsigmasq_c_history = np.expand_dims(np.zeros_like(self.model.logsigmasq_c), axis = 2)
        
        self.mu = []
        self.logsigmasq = []
        self.train_labels = []
        
        self.mu_history = np.expand_dims(np.zeros((len(self.train_data.dataset), self.model.mu_c.shape[1])), axis = 2)
        self.logsigmasq_history = np.expand_dims(np.zeros((len(self.train_data.dataset), self.model.logsigmasq_c.shape[1])), axis = 2)
        self.min_val_loss = torch.inf
        self.min_train_loss = torch.inf
        
        self._check_run()
        
        configure(self.run_path)
            
    def _check_run(self):
        
        logs_path = self.logs_path
        
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
            
        run_list = [item for item in os.listdir(logs_path) if item.split('_')[0] == 'run']
        if run_list:
            idx = [int(item.split('_')[-1]) for item in run_list]
            idx.sort()
            last_run = idx[-1]
            
            #If the epochs ran in the last run are less than the total epochs then load the model and restart
            snapshot_path = os.path.join(logs_path, f'run_{last_run}/snapshot.pt')
            if self.restart and os.path.exists(snapshot_path):
                
                if self.cuda:
                    loc = f"cuda:{self.gpu_id}"
                    snapshot = torch.load(snapshot_path, map_location=loc)
                else:
                    snapshot = torch.load(snapshot_path)
                
                if snapshot['EPOCHS_RUN'] < snapshot['TOTAL_EPOCHS']-1:
                    print("Loading snapshot")
                    self.current_run = last_run
                    self.run_path = os.path.join(logs_path, f'run_{self.current_run}')
                    self._load_snapshot(snapshot_path)
                    self._load_variables(os.path.join(logs_path, f'run_{last_run}'))
                else:
                    self.current_run = last_run + 1
                    self.run_path = os.path.join(logs_path, f'run_{self.current_run}')
                    os.makedirs(self.run_path)
            else:
                self.current_run = last_run + 1
                self.run_path = os.path.join(logs_path, f'run_{self.current_run}')
                os.makedirs(self.run_path)            
        else:
            self.current_run = 0
            self.run_path = os.path.join(logs_path, f'run_{self.current_run}')
            os.makedirs(self.run_path)   

    def _load_snapshot(self, snapshot_path):
        if self.cuda:
            loc = f"cuda:{self.gpu_id}"
            snapshot = torch.load(snapshot_path, map_location=loc)
        else:
            snapshot = torch.load(snapshot_path)
            
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.model.eval()
        self.epochs_run = snapshot["EPOCHS_RUN"]
        
        print(f"Resuming training from run_{self.current_run} snapshot at Epoch {self.epochs_run}")
        
    def _load_variables(self, variables_path):
        
        
        self.epoch_list = np.load(os.path.join(variables_path,'train_loss_epoch.npy'))
        
        self.train_loss = np.load(os.path.join(variables_path,'train_loss_epoch.npy'))
        self.train_rec_loss = np.load(os.path.join(variables_path,'train_rec_loss_epoch.npy'))
        self.train_reg_loss = np.load(os.path.join(variables_path,'train_reg_loss_epoch.npy'))
        self.train_entr_loss = np.load(os.path.join(variables_path,'train_entr_loss_epoch.npy'))
        
        self.val_loss = np.load(os.path.join(variables_path,'val_loss_epoch.npy'))
        self.val_rec_loss = np.load(os.path.join(variables_path,'val_rec_loss_epoch.npy'))
        self.val_reg_loss = np.load(os.path.join(variables_path,'val_reg_loss_epoch.npy'))
        self.val_entr_loss = np.load(os.path.join(variables_path,'val_entr_loss_epoch.npy'))         
        
        self.pi_history = np.load(os.path.join(variables_path,'pi_epoch.npy'))
        self.mu_c_history = np.load(os.path.join(variables_path,'mu_c_epoch.npy'))
        self.logsigmasq_c_history = np.load(os.path.join(variables_path,'logsigmasq_c_epoch.npy'))
        self.mu_history = np.load(os.path.join(variables_path,'mu_epoch.npy'))
        self.logsigmasq_history = np.load(os.path.join(variables_path,'logsigmasq_epoch.npy'))
        
    #Run train batches (backprop activated)
    def _run_batch_train(self, source, targets):
        
        self.optimizer.zero_grad()
        
        loss, rec, reg, entr = self.model.loss(source, 'train')
        
        loss.backward()
        
        self.optimizer.step()
        
        return loss, rec, reg, entr
        
    #Run val batches (backprop deactivated)
    def _run_batch_val(self, source, targets):
                
        loss, rec, reg, entr = self.model.loss(source, 'val')
                        
        return loss, rec, reg, entr

    #Run epoch over entire dataset (training and validation)
    def _run_epoch(self, epoch, dataset, mode='train'):
        
        b_sz = len(next(iter(dataset))[0])
        if self.cuda:
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(dataset)}")
            self.dataset.sampler.set_epoch(epoch)
        #else:
            #print(f"Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(dataset)}")
            
        avg_train_loss = 0
        avg_rec_train_loss = 0
        avg_reg_train_loss = 0
        avg_entr_train_loss = 0
        avg_val_loss = 0
        avg_rec_val_loss = 0
        avg_reg_val_loss = 0
        avg_entr_val_loss = 0
        train_batches = 0
        val_batches = 0
        
        #Batch loop
        for step, data in enumerate(dataset):

            source, targets = data
            
            if self.cuda:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                
            if mode == 'train':
                loss, rec, reg, entr = self._run_batch_train(source, targets)

                avg_train_loss += loss.item()
                avg_rec_train_loss += rec.item()
                avg_reg_train_loss += reg.item()
                avg_entr_train_loss += entr.item()
                
                train_batches += 1
                
                if epoch == 0:
                    self.train_labels.append(targets.numpy())
                
            else:
                loss, rec, reg, entr = self._run_batch_val(source, targets)
            
                avg_val_loss += loss.item()
                avg_rec_val_loss += rec.item()
                avg_reg_val_loss += reg.item()
                avg_entr_val_loss += entr.item()
                
                val_batches += 1
        #Average over number of batches     
        if mode == 'train':
      
            avg_train_loss /= train_batches
            
            avg_rec_train_loss /= train_batches
            avg_reg_train_loss /= train_batches
            avg_entr_train_loss /= train_batches
            
            if epoch == 0:
                self.train_labels = np.concatenate(self.train_labels, axis = 0)
                np.save(self.run_path + '/train_labels',self.train_labels)  
                  
            return avg_train_loss, avg_rec_train_loss, avg_reg_train_loss, avg_entr_train_loss
             
        else: 
            avg_val_loss /= val_batches

            avg_rec_val_loss /= val_batches
            avg_reg_val_loss /= val_batches
            avg_entr_val_loss /= val_batches
            
            return avg_val_loss, avg_rec_val_loss, avg_reg_val_loss, avg_entr_val_loss
            
    def _save_snapshot(self, epoch):
    
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": epoch,
            "TOTAL_EPOCHS": self.total_epochs,
        }
        #Save snapshot at specific epoch
        snapshot_path = os.path.join(self.run_path, f'snapshot_epoch_{epoch}.pt')
        torch.save(snapshot, snapshot_path)
        #Overwrite last snapshot 
        torch.save(snapshot, os.path.join(self.run_path, f'snapshot_last.pt'))
        print(f"\nEpoch {epoch} | Training snapshot saved at {snapshot_path}")

    #Main method to train the model
    def train(self):
    
                
        #Prepare data
        tr_samples = len(self.train_data.dataset)
        tr_datasize = [(i+1)*int(tr_samples/self.total_episodes) for i in range(self.total_episodes)]
        tr_full_data = self.train_data.dataset
        num_workers = self.train_data.num_workers
        
        val_samples = len(self.val_data.dataset)
        val_datasize = [(i+1)*int(val_samples/self.total_episodes) for i in range(self.total_episodes)]
        val_full_data = self.val_data.dataset
    
        for episode in range(self.total_episodes):

            tr_data = tr_full_data[0:tr_datasize[episode]]
            episode_tr_dataset = DataLoader(tr_data, batch_size=tr_datasize[episode], shuffle=False, num_workers = num_workers)
            
            val_data = val_full_data[0:val_datasize[episode]]
            episode_tr_dataset = DataLoader(val_data, batch_size=val_datasize[episode], shuffle=False, num_workers = num_workers)
            
            self.train_labels = []
            self.mu = []
            self.logsigmasq = []
            max_epochs = self.total_epochs
            
            
        
            for epoch in range(self.epochs_run, max_epochs):
            
                meta_epoch = episode*max_epochs + epoch
                self.epoch_list = np.append(self.epoch_list,meta_epoch)
                np.save(self.run_path + '/epochs',self.epoch_list)
            
                self.model.encoder.train()
                self.model.decoder.train()
                
                #Reset running weights
                self.model.params['hist_weights'] = torch.zeros_like(self.model.params['hist_weights']).clone().detach()
                self.model.params['hist_mu_c'] = torch.zeros_like(self.model.params['hist_mu_c']).clone().detach()
                self.model.params['hist_logsigmasq_c'] = torch.zeros_like(self.model.params['hist_logsigmasq_c']).clone().detach()
            
                #TRAINING
                train_loss, train_rec, train_reg, train_entr = self._run_epoch(epoch, self.train_data, 'train')
            
                #if self.min_train_loss > train_loss and episode > 1000:
                #    self.min_train_loss = train_loss
                #    self.save_flag = True

                self.train_loss = np.append(self.train_loss,train_loss)
                self.train_rec_loss = np.append(self.train_rec_loss,train_rec)
                self.train_reg_loss = np.append(self.train_reg_loss,train_reg)
                self.train_entr_loss = np.append(self.train_entr_loss,train_entr)
            
                np.save(self.run_path + '/train_loss_epoch',self.train_loss)
                np.save(self.run_path + '/train_rec_loss_epoch',self.train_rec_loss)
                np.save(self.run_path + '/train_reg_loss_epoch',self.train_reg_loss)
                np.save(self.run_path + '/train_entr_loss_epoch',self.train_entr_loss)
            
                log_value('train_loss_epoch', train_loss, epoch)
                log_value('reconstruction_train_loss_epoch', train_rec, epoch)
                log_value('regularization_train_loss_epoch', train_reg, epoch)
                log_value('entropy_train_loss_epoch', train_entr, epoch)
            
                self.pi_history = np.concatenate((self.pi_history,torch.unsqueeze(self.model.params['pi_c'], dim=1).detach().numpy()), axis = 1)
                self.mu_c_history = np.concatenate((self.mu_c_history,np.expand_dims(self.model.mu_c.detach(), axis = 2)), axis = 2)
                self.logsigmasq_c_history = np.concatenate((self.logsigmasq_c_history,np.expand_dims(self.model.logsigmasq_c.detach(), axis = 2)), axis = 2)
            
                #Encode input data onto latent space
                with torch.no_grad():
                    self.model.encoder.eval()
                    self.model.decoder.eval()
        
                    #Concatenate all batches
                    x_list = [torch.cat([source for source, target in self.train_data], axis = 0)]
        
                    mu, logsigmasq = self.model._encoder_step(x_list, self.model.encoder, self.model.encoder)   

                self.mu_history = np.concatenate((self.mu_history,np.expand_dims(mu, axis = 2)), axis = 2)
                self.logsigmasq_history = np.concatenate((self.logsigmasq_history,np.expand_dims(logsigmasq, axis = 2)), axis = 2)
                self.mu = []
                self.logsigmasq = []
            
                np.save(self.run_path + '/pi_epoch',self.pi_history)
                np.save(self.run_path + '/mu_c_epoch',self.mu_c_history)
                np.save(self.run_path + '/logsigmasq_c_epoch',self.logsigmasq_c_history)
                np.save(self.run_path + '/mu_epoch',self.mu_history)
                np.save(self.run_path + '/logsigmasq_epoch',self.logsigmasq_history)            
            
                if self.cuda:
                    if self.gpu_id == 0 and (epoch % self.save_every == 0 or epoch == self.total_epochs - 1):
                        self._save_snapshot(epoch)
                else:
                    if (epoch % self.save_every == 0 or epoch == self.total_epochs - 1):
                        self._save_snapshot(epoch)
                    
                #if self.save_flag:
                    #self._save_snapshot(epoch)
                    #self.save_flag = False

                #VALIDATION
                with torch.no_grad():
                    #print('\nValidation:')
                    
                    self.model.encoder.eval()
                    self.model.decoder.eval()

                    val_loss, val_rec, val_reg, val_entr = self._run_epoch(epoch, self.val_data, 'val')

                    self.val_loss = np.append(self.val_loss,val_loss)
                    self.val_rec_loss = np.append(self.val_rec_loss,val_rec)
                    self.val_reg_loss = np.append(self.val_reg_loss,val_reg)
                    self.val_entr_loss = np.append(self.val_entr_loss,val_entr)
                
                    np.save(self.run_path + '/val_loss_epoch',self.val_loss)
                    np.save(self.run_path + '/val_rec_loss_epoch',self.val_rec_loss)
                    np.save(self.run_path + '/val_reg_loss_epoch',self.val_reg_loss)
                    np.save(self.run_path + '/val_entr_loss_epoch',self.val_entr_loss)
                
                    log_value('val_loss_epoch', val_loss, epoch)
                    log_value('reconstruction_val_loss_epoch', val_rec, epoch)
                    log_value('regularization_val_loss_epoch', val_reg, epoch)
                    log_value('entropy_val_loss_epoch', val_entr, epoch)

                print(f'Episode {episode} ({tr_datasize[episode]} training / {val_datasize[episode]} validation samples)| Epoch {meta_epoch} | Train loss: {train_loss} | Val loss: {val_loss}')

