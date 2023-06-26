import argparse
import logging
from os.path import join

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from rdkit import Chem, RDLogger
from yaml import Dumper, Loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import numpy as np
import torch
import torch.nn as nn
from data_utils import dataset_graph, dataset_graph_finetune
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from train_module.Pretrain_Graph import Pretrain_Graph

"""
This script is used to train the pretrained base Uni-Rxn model
Run:
    python trainer_pretraining_graph_pl.py --config_path config/
    The hyperparameters are set in the config folder
Output:
    The model checkpoints/runtime tensorboard/configs will be saved in the pl_logs folder
    tensorboard --logdir pl_logs to visualize the training process

"""

parser = argparse.ArgumentParser()
parser.add_argument('--config_path',
                    type=str,
                    default='./config')


args = parser.parse_args()
cfg = edict({
    'dataset_pretrain':
    yaml.load(open(join(args.config_path, 'dataset/pretrain.yaml')),
                Loader=Loader),
    'model':
    yaml.load(open(join(args.config_path, 'model/pretrain_graph.yaml')),
              Loader=Loader),
    'optim':
    yaml.load(open(join(args.config_path, 'optim/adam.yaml')), Loader=Loader),
    'scheduler':
    yaml.load(open(join(args.config_path, 'scheduler/step.yaml')),
              Loader=Loader),
    'trainer':
    yaml.load(open(join(args.config_path, 'trainer/default.yaml')),
              Loader=Loader)
})

class pretrain_data_module(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        whole_dataset = dataset_graph.reaction_pretrain_dataset(self.cfg.dataset_pretrain.dataset.dataset_loc, self.cfg.dataset_pretrain.dataset.vocab_loc)
        train_size = int(0.9*len(whole_dataset))
        valid_size = len(whole_dataset) - train_size
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(whole_dataset,[train_size, valid_size])
        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg.dataset_pretrain.loader, collate_fn=dataset_graph.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, **self.cfg.dataset_pretrain.loader, collate_fn=dataset_graph.collate_fn)
        

if __name__ == '__main__':
    tb_logger = TensorBoardLogger(cfg.trainer.default_root_dir, name='pretrain_graph_remove_overlap(big lambda)', flush_secs=120)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(**cfg.trainer, logger=tb_logger, callbacks=[EarlyStopping(monitor="valid_loss",patience=8), lr_monitor])
    
    model = Pretrain_Graph(cfg)
    data_module = pretrain_data_module(cfg)
    trainer.fit(model, data_module)