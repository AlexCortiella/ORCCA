###### YOLO NAS TRAINER #####

import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torchvision
from torchvision.utils import make_grid
from torchvision.io import read_image
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import matplotlib.image as mpimg

import os

import requests
from PIL import Image
from tqdm import tqdm as tqdm

from super_gradients.training import Trainer, dataloaders, models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val
)
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback
)

##############################
###### CONFIGURATION #########
##############################
class config:
    #trainer params
    CHECKPOINT_DIR = 'path/to/checkpoint/dir' #specify the path you want to save checkpoints to
    EXPERIMENT_NAME = 'test' #specify the experiment name

    #dataset params
    DATA_DIR = './data/mars_rover' #parent directory to where data lives

    TRAIN_IMAGES_DIR = 'train/images' #child dir of DATA_DIR where train images are
    TRAIN_LABELS_DIR = 'train/labels' #child dir of DATA_DIR where train labels are

    VAL_IMAGES_DIR = 'valid/images' #child dir of DATA_DIR where validation images are
    VAL_LABELS_DIR = 'valid/labels' #child dir of DATA_DIR where validation labels are

    # if you have a test set
    TEST_IMAGES_DIR = 'test/images' #child dir of DATA_DIR where test images are
    TEST_LABELS_DIR = 'test/labels' #child dir of DATA_DIR where test labels are

    CLASSES = ['rock', 'rover'] #what class names do you have

    NUM_CLASSES = len(CLASSES)

    #dataloader params - you can add whatever PyTorch dataloader params you have
    #could be different across train, val, and test
    DATALOADER_PARAMS={
    'batch_size':4,
    'num_workers':2
    }
    # model params
    MODEL_NAME = 'yolo_nas_l' # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
    PRETRAINED_WEIGHTS = 'coco' #only one option here: coco

##############################
###### DATA LOADERS ##########
##############################
train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': config.DATA_DIR,
        'images_dir': config.TRAIN_IMAGES_DIR,
        'labels_dir': config.TRAIN_LABELS_DIR,
        'classes': config.CLASSES
    },
    dataloader_params=config.DATALOADER_PARAMS
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': config.DATA_DIR,
        'images_dir': config.VAL_IMAGES_DIR,
        'labels_dir': config.VAL_LABELS_DIR,
        'classes': config.CLASSES
    },
    dataloader_params=config.DATALOADER_PARAMS
)

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': config.DATA_DIR,
        'images_dir': config.TEST_IMAGES_DIR,
        'labels_dir': config.TEST_LABELS_DIR,
        'classes': config.CLASSES
    },
    dataloader_params=config.DATALOADER_PARAMS
)

    
   
##############################
#### TRAINING PARAMETERS #####
##############################    
train_params = {
    # ENABLING SILENT MODE
    "gpus": 1,
    "average_best_models":True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    # ONLY TRAINING FOR 10 EPOCHS FOR THIS EXAMPLE NOTEBOOK
    "max_epochs": 10,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        # NOTE: num_classes needs to be defined here
        num_classes=config.NUM_CLASSES,
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            # NOTE: num_classes needs to be defined here
            num_cls=config.NUM_CLASSES,
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50'
}    




if __name__ == "__main__":

	from super_gradients.training import Trainer, dataloaders, models
	from super_gradients.training.dataloaders.dataloaders import (
	    coco_detection_yolo_format_train, coco_detection_yolo_format_val
	)
	from super_gradients.training.utils.distributed_training_utils import setup_device
	from super_gradients.training.losses import PPYoloELoss
	from super_gradients.training.metrics import DetectionMetrics_050
	from super_gradients.training.models.detection_models.pp_yolo_e import (
	    PPYoloEPostPredictionCallback
)
	# Launch DDP on 2 GPUs'
	#setup_device(num_gpus=train_params['gpus']) # Equivalent to: setup_device(multi_gpu='DDP', num_gpus=2)

	trainer = Trainer(experiment_name=config.EXPERIMENT_NAME, ckpt_root_dir=config.CHECKPOINT_DIR)
	model = models.get(config.MODEL_NAME, 
		           num_classes=config.NUM_CLASSES, 
		           pretrained_weights=config.PRETRAINED_WEIGHTS
		           )
	trainer.train(model=model, 
		      training_params=train_params, 
		      train_loader=train_data, 
		      valid_loader=val_data)
