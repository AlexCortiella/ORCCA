#MAIN TRAIN

def main(cfg):
    
    logs_path = cfg.logs_path
    snapshot_path = cfg.snapshot_path
    generator = Generator(cfg, n_samples, logs_path, snapshot_path)
    print('Generator initialized...')
    samples, run_path = generator.generate()
    print('Samples generated!')
    return samples, run_path


if __name__ == "__main__":
    
    import argparse
    import pprint
    import yaml
    from core.model import *
    from core.generator import Generator
    from core.data import *
    from core.utils import *
    
    parser = argparse.ArgumentParser(description='Input configuration file')
    parser.add_argument('-f','--config', type=str,
                          help='Configuration filepath that contains model parameters',
                          default = './config_file_multigpu.yaml')
    parser.add_argument('-s','--samples', type=int,
                          help='Number of samples to generate',
                          default = './config_file_multigpu.yaml')                      
    parser.add_argument('-j','--jobid', type=str,
                          help='JOB ID',
                          default = '000000')

    args = parser.parse_args()

    config_filepath = args.config
    n_samples = args.samples
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = dic2struc(cfg)
    
    samples, run_path = main(cfg)
    
    gen_path = os.path.join(run_path, "generated_samples.json")
    
    print('Saving generated samples...')
    with open(gen_path, "w") as outfile:
        json.dump(samples, outfile)
    
    
