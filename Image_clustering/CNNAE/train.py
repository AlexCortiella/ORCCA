#MAIN TRAIN
import argparse
import pprint
import yaml
import json
import os
from core.model import *
from core.data import *
from core.trainer import *
from core.utils import *

# Pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def main(cfg):

    train_loader, valid_loader, test_loader, _, optimizer = load_train_objs(cfg)
    
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(cfg.logs_path,f"cats_and_dogs_{cfg.latent_dim}"),  
                         accelerator=cfg.device,
                         devices=cfg.num_devices,
                         max_epochs=cfg.total_epochs,
                         callbacks=[ModelCheckpoint(every_n_epochs=50),
                                    LearningRateMonitor("epoch")])
    print("HELLO")
#    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
#    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = cfg.pretrained_path
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = CNNAE.load_from_checkpoint(pretrained_filename)
    else:
        model = CNNAE(cfg)
        
    print("Starting training...")
    trainer.fit(model, train_loader, valid_loader)
        
    # Test best model on validation and test set
    valid_result = trainer.test(model, valid_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": valid_result}
    
    cfg_dict = cfg.__dict__
    
#    with open(os.path.join(trainer.run_path, 'config_file.json'),'w')as outfile:
#        json.dump(cfg_dict, outfile)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Input configuration file')
    parser.add_argument('-f','--config', type=str,
                          help='Configuration filepath that contains model parameters',
                          default = './config_file_multigpu.yaml')

    args = parser.parse_args()

    config_filepath = args.config

    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = dic2struc(cfg)
    
    #Get number of gpus
    #world_size = torch.cuda.device_count() 
    main(cfg)
   
