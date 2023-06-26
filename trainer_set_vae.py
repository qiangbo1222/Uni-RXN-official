import argparse
import logging
import os
import pickle
import time
from os.path import join

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from yaml import Dumper, Loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data_utils.dataset_graph_generation import chains_dataset, collate_fn
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from train_module.CATH_vae import CATH

"""
This script is used to train the generative model Uni-Rxn  that is based on the pretrained base Uni-Rxn model
Run:
    rewrite the config/model/cath_vae.yaml: "pretrained_path" to your pretrained model path (e.g. pl_logs/pretrain_graph/version_0/checkpoints/epoch=0-step=0.ckpt)
    python trainer_set_vae.py --config_path config/
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
    'dataset':
    yaml.load(open(join(args.config_path, 'dataset/chains.yaml')),
              Loader=Loader),
    'model':
    yaml.load(open(join(args.config_path, 'model/cath_vae.yaml')),
              Loader=Loader),
    'optim':
    yaml.load(open(join(args.config_path, 'optim/adam.yaml')), Loader=Loader),
    'trainer':
    yaml.load(open(join(args.config_path, 'trainer/default.yaml')),
              Loader=Loader),
    'checkpoint':
    yaml.load(open(join(args.config_path, 'checkpoint/default.yaml')),
                Loader=Loader)
})

class chains_data_module(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        #set random seed
        torch.manual_seed(2023)
        whole_dataset = chains_dataset(self.cfg.dataset.dataset_loc, self.cfg.dataset.vocab_loc)
        train_size = int(0.8*len(whole_dataset))
        valid_size = int(0.1*len(whole_dataset))
        test_size = len(whole_dataset) - train_size - valid_size
        self.train_dataset, self.valid_dataset, self.test_dataset = torch.utils.data.random_split(whole_dataset,[train_size, valid_size, test_size])

    def train_dataloader(self):     
        return DataLoader(self.train_dataset, **self.cfg.dataset.loader, collate_fn=collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, **self.cfg.dataset.loader, collate_fn=collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.cfg.dataset.loader, collate_fn=collate_fn)


checkpoint_func = ModelCheckpoint(**cfg.checkpoint)
tb_logger = TensorBoardLogger(cfg.trainer.default_root_dir, name='set_gen_vae', flush_secs=120)


lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = Trainer(**cfg.trainer, logger=tb_logger, callbacks=[checkpoint_func, lr_monitor])
model = CATH(cfg)
trainer.fit(model, chains_data_module(cfg))
trainer.test(model, chains_data_module(cfg))
