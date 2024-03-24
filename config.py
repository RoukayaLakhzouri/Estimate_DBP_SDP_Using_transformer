# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 22:30:23 2024

@author: user
"""

from pathlib import Path
# config is used in case of crashing we will resume the training 
def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,#the number of times the entire dataset is passed forward and backward through the model
        "lr": 10**-4,#learning rate
        "d_model": 512,
        "datasource": 'custom_dataset',
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}" #the folder path where the model weights will be saved
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])