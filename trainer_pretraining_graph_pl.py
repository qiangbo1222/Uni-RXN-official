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
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data_utils import dataset_graph, dataset_graph_finetune
from train_module.Pretrain_Graph import Pretrain_Graph

parser = argparse.ArgumentParser()
parser.add_argument('--config_path',
                    type=str,
                    default='./config')
parser.add_argument('--stage',
                    type=str,
                    default='pretrain')
parser.add_argument('--pretrain_ckpt',
                    type=str,
                    default='./pl_logs/pretrain_graph/version_46/checkpoints/epoch=79-step=498240.ckpt')

args = parser.parse_args()
cfg = edict({
    'dataset_finetune':
    yaml.load(open(join(args.config_path, 'dataset/finetune.yaml')),
              Loader=Loader),
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
        
class pretrain_data_finetune_module(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        whole_dataset = dataset_graph_finetune.reaction_pretrain_dataset(self.cfg.dataset_finetune.dataset.dataset_loc, self.cfg.dataset_finetune.dataset.vocab_loc)
        train_size = int(0.9*len(whole_dataset))
        valid_size = len(whole_dataset) - train_size
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(whole_dataset,[train_size, valid_size])
        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.cfg.dataset_finetune.loader, shuffle=True, collate_fn=dataset_graph_finetune.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, **self.cfg.dataset_finetune.loader, shuffle=False, collate_fn=dataset_graph_finetune.collate_fn)
        
if __name__ == '__main__':
    tb_logger = TensorBoardLogger(cfg.trainer.default_root_dir, name='pretrain_graph', flush_secs=120)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    if args.stage == 'pretrain':
        trainer = Trainer(**cfg.trainer, logger=tb_logger, callbacks=[EarlyStopping(monitor="valid_loss"), lr_monitor])
        model = Pretrain_Graph(cfg)
        data_module = pretrain_data_module(cfg)
        trainer.fit(model, data_module)
    elif args.stage == 'finetune':
        trainer = Trainer(**cfg.trainer, logger=tb_logger, callbacks=[EarlyStopping(monitor="valid_accuracy", mode='max'), lr_monitor])
        model = Pretrain_Graph(cfg, stage='finetune')
        model = model.load_from_checkpoint(args.pretrain_ckpt)
        trainer.fit(model, pretrain_data_finetune_module(cfg))
