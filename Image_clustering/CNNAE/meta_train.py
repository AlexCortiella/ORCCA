#MAIN TRAIN

def main(cfg):
    
    if cfg.cuda:
        ddp_setup()
    
    train_data, val_data, model, optimizer = load_train_objs(cfg)
    
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    trainer = Trainer(model, train_data, val_data, optimizer, cfg.save_every, cfg.total_epochs, cfg.total_episodes, cfg.logs_path, cfg.restart, cfg.cuda)
    trainer.train()
    
    cfg_dict = cfg.__dict__
    
    with open(os.path.join(trainer.run_path, 'config_file.json'),'w')as outfile:
        json.dump(cfg_dict, outfile)
    
    
    if cfg.cuda:
        destroy_process_group()


if __name__ == "__main__":
    
    import argparse
    import pprint
    import yaml
    import json
    import os
    from core.model import *
    from core.data import *
    from core.meta_trainer import *
    from core.utils import *
    
    parser = argparse.ArgumentParser(description='Input configuration file')
    parser.add_argument('-f','--config', type=str,
                          help='Configuration filepath that contains model parameters',
                          default = './config_file_multigpu.yaml')
    parser.add_argument('-j','--jobid', type=str,
                          help='JOB ID',
                          default = '000000')

    args = parser.parse_args()

    config_filepath = args.config

    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = dic2struc(cfg)
    
    main(cfg)
