# PREDICTOR

import torch
import torch.nn.functional as F
import torch.nn as nn
from core.model import GMVAE
import os
import numpy as np
import json

class Generator():
   
    def __init__(self, cfg, n_samples, logs_path, snapshot_path = None):
       
        self.model = GMVAE(cfg)
        self.n_samples = n_samples
        self.logs_path = logs_path
        self.run_path = None
       
        if snapshot_path is None:
        
            run_list = [item for item in os.listdir(self.logs_path) if item.split('_')[0] == 'run']
            if run_list:
                idx = [int(item.split('_')[-1]) for item in run_list]
                idx.sort()
                last_run = idx[-1]
            
                #If the epochs ran in the last run are less than the total epochs then load the model and restart
                self.run_path = os.path.join(logs_path, f'run_{last_run}')
                self.snapshot_path = os.path.join(logs_path, f'run_{last_run}/snapshot_last.pt')
            elif cfg.snapshot_path:
                self.snapshot_path = cfg.snapshot_path
                self.run_path = os.path.join(*(cfg.snapshot_path).split('/')[0:-1]) 
            else:
                print('Snapshot path not specified.')
        else:
            self.snapshot_path = snapshot_path
            self.run_path = os.path.join(*(cfg.snapshot_path).split('/')[0:-1]) 
           
    def _load_inference_objs(self):
       
        #Load model
        snapshot = torch.load(self.snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.model.eval()
        print(f'Model loaded from: {self.snapshot_path}')
        
    def generate(self, n_samples = None):
       
        if n_samples:
            self.n_samples = n_samples
            
        self._load_inference_objs()
                
        samples = {}
        
        C, Z, X = self.model.generate(self.n_samples)
        
        for s in range(self.n_samples):
            samples[f'sample {s}'] = {}
            samples[f'sample {s}']['data x'] = X[:,s].tolist()
            samples[f'sample {s}']['data z'] = Z[:,s].tolist()
            samples[f'sample {s}']['label'] = int(C[s])
            
        return samples, self.run_path

            
