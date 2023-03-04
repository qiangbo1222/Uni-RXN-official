import imp
import logging
import math
from pycm import *
import pickle
import numpy as np
from typing import Any, Dict

import dataset.syn_dag_from_qb.chem_utils_for_reactant as chem_utils
import dataset.syn_dag_from_qb.parser_selfies as parser_selfies
import info_nce
import rdkit.Chem as Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils.dataset_graph import graph_collator
from data_utils.parser import mol_to_graph, preprocess_item
from model.pretraining_encoder_graph import (MLP_head, Pos_embeddings,
                                             agg_encoder, graph_encoder,
                                             utter_encoder)
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Pretrain_Graph(LightningModule):
    def __init__(self,cfg: Dict[str,Any], ckpt_path):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.react_graph_model = graph_encoder(cfg.model.graph_encoder)
        self.react_graph_model_cross = graph_encoder(cfg.model.graph_encoder_cross, cross=True)
        self.prod_graph_model = graph_encoder(cfg.model.graph_encoder)

        pos_embed = Pos_embeddings(**cfg.model.pos_embed)
        self.reagent_model = utter_encoder(**cfg.model.utter_encoder, embedder=pos_embed)
        self.reagent_encoder = agg_encoder(**cfg.model.reagent_encoder)

        self.center_predictor = nn.Sequential(MLP_head(**cfg.model.center_predictor), 
                                                nn.Sigmoid())
        self.input_head = MLP_head(**cfg.model.graph_head)
        self.reactants_head = MLP_head(**cfg.model.graph_head)
        self.graph_head = MLP_head(**cfg.model.graph_head)
        self.reagent_head = MLP_head(**cfg.model.reagent_head)
        #load pretrain model
        self.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
        print("loading checkpoints")

        self.finetune_head_yield = MLP_head(**cfg.model.finetune_head_yield, first_dropout=False)
        
    

    def forward(self, batch):
        return self.finetune(batch)
    
    
    def finetune(self, batch):
        graph_reactants_encoding = self.react_graph_model(batch['graph_reactants'])
        graph_inputs_encoding = self.react_graph_model(batch['graph_input'])
        graph_prod_encoding = self.prod_graph_model(batch['graph_prod'])

        bs, reactant_num_max, reactant_len = batch['utter'].shape
        reagent_num = batch['reagent_num']

        utter = batch['utter'].reshape(bs*reactant_num_max, -1)
        utter_encoding = self.reagent_model(utter)
        utter_encoding = utter_encoding.reshape(bs, reactant_num_max, reactant_len, -1)
        utter_encoding = utter_encoding[:, :, 0, :]
        reagent_encoding = self.reagent_encoder(utter_encoding, reagent_num)

        rxn_rep = torch.cat([graph_inputs_encoding[:, 0, :], torch.sum(graph_reactants_encoding, dim=1), graph_prod_encoding[:,0,:], reagent_encoding], dim=-1)

        class_predict = self.finetune_head_yield(rxn_rep).squeeze(-1)
        loss = nn.CrossEntropyLoss()(class_predict, batch['labels'])
        accuracy = (class_predict.argmax(dim=-1) == batch['labels']).float().mean()
        
        class_predict = torch.argmax(nn.LogSoftmax(dim=-1)(class_predict), dim=-1).cpu().detach().numpy()
        cm = ConfusionMatrix(actual_vector=batch['labels'].cpu().detach().numpy(), predict_vector=class_predict)
        cen = cm.overall_stat['Overall CEN']
        mcc = cm.overall_stat['Overall MCC']


        return {'loss': loss, 'accuracy': accuracy, 'mcc': torch.tensor(mcc).type_as(loss), 'cen': torch.tensor(cen).type_as(loss)}
    
    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('train_loss' , loss["loss"], on_epoch=True, prog_bar=True)
        self.log('train_accuracy', loss["accuracy"], on_epoch=True, prog_bar=True)
        return loss

    
    def configure_optimizers(self):
        tune_param = list(self.finetune_head_yield.parameters())
        base_param = list(self.react_graph_model.parameters())
        optimizer = torch.optim.AdamW([{'params': tune_param, 'lr': self.cfg.optim.lr * 5},
                                        {'params': base_param, 'lr': self.cfg.optim.lr}], lr=self.cfg.optim.lr * 1, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg.scheduler.gamma)
        scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

        return [optimizer], [scheduler]
    
    def _set_num_training_steps(self, scheduler_cfg):
        if "num_training_steps" in scheduler_cfg:
            scheduler_cfg = dict(scheduler_cfg)
            if self.global_rank == 0:
                logger.info("Computing number of training steps...")
                num_training_steps = [self.num_training_steps()]
            else:
                num_training_steps = [0]
            torch.distributed.broadcast_object_list(
                num_training_steps,
                0,
                group=torch.distributed.group.WORLD,
            )
            scheduler_cfg["num_training_steps"] = num_training_steps[0]
            logger.info(
                f"Training steps: {scheduler_cfg['num_training_steps']}"
            )
        return scheduler_cfg

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.num_training_batches != float("inf"):
            dataset_size = self.trainer.num_training_batches
        else:
            dataset_size = len(
                self.trainer._data_connector._train_dataloader_source.dataloader()
            )

        if (
            isinstance(self.trainer.limit_train_batches, int)
            and self.trainer.limit_train_batches > 0
        ):
            dataset_size = min(dataset_size, self.trainer.limit_train_batches)
        else:
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)

        accelerator_connector = self.trainer._accelerator_connector
        if accelerator_connector.use_ddp2 or accelerator_connector.use_dp:
            effective_gpus = 1
        else:
            effective_gpus = self.trainer.devices
            if effective_gpus < 0:
                effective_gpus = torch.cuda.device_count()

        effective_devices = effective_gpus * self.trainer.num_nodes
        effective_batch_size = (
            self.trainer.accumulate_grad_batches * effective_devices
        )
        max_estimated_steps = (
            math.ceil(dataset_size // effective_batch_size)
            * self.trainer.max_epochs
        )
        logger.info(
            f"{max_estimated_steps} = {dataset_size} // "
            f"({effective_gpus} * {self.trainer.num_nodes} * "
            f"{self.trainer.accumulate_grad_batches}) "
            f"* {self.trainer.max_epochs}"
        )

        max_estimated_steps = (
            min(max_estimated_steps, self.trainer.max_steps)
            if self.trainer.max_steps and self.trainer.max_steps > 0
            else max_estimated_steps
        )
        return max_estimated_steps
    
    def _compute_metrics(self, result):
        return {k: l.mean() for k, l in result.items()}
    
    def validation_step(self, batch, batch_idx):
        result = self.forward(batch)
        return result
    

    def validation_epoch_end(self, result):
        loss = self._compute_metrics(self._gather_result(result))
        for k, l in loss.items():
            self.log('valid_' + k, l, on_epoch=True, prog_bar=True, batch_size=8)

    def test_step(self, batch, batch_idx):
        result = self.forward(batch)
        return result
    
    def test_epoch_end(self, result):
        loss = self._compute_metrics(self._gather_result(result))
        for k, l in loss.items():
            self.log('test_' + k, l, on_epoch=True, prog_bar=True, batch_size=8)
    
    def _gather_result(self, result):
        # collect steps
        result = {
            key: torch.cat([x[key] for x in result])
            if len(result[0][key].shape) > 0
            else torch.tensor([x[key] for x in result]).to(result[0][key])
            for key in result[0].keys()
        }
        # collect machines
        #result = {
        #    key: torch.cat(list(self.all_gather(result[key])))
        #    for key in result.keys()
        #}
        return result
