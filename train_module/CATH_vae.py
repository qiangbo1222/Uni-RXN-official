import copy
import logging
import math
import os
import pickle
import random
from typing import Any, Dict

import dataset.syn_dag_from_qb.chem_utils_for_reactant as chem_utils
import dataset.syn_dag_from_qb.parser_selfies as parser_selfies
import info_nce
import rdkit
import torch
from data_utils.dataset_graph import graph_collator
from data_utils.parser import mol_to_graph, preprocess_item
from model.Set_modules_vae import *
from pytorch_lightning import LightningModule
from rdkit import Chem
#from model.Set_modules import CATH
from torch import nn
from torch.nn import functional as F

#from train_module.get_pretrained import Main_encoder

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CATH(LightningModule):
    def __init__(self,cfg: Dict[str,Any]):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.ablation = cfg.model.ablation

        self.path_encoder = Path_encoder(cfg.model)
        #Only run when use pretraining weights (remember to cancel when inferencing)
        self.path_encoder.load_checkpoint(cfg.model.pretrained_path)
        self.target_encoder = Target_encoder(cfg.model)
        #Only run when use pretraining weights (remember to cancel when inferencing)
        self.target_encoder.load_checkpoint(cfg.model.pretrained_path)
        self.prior_net = Dis_net(**cfg.model.prior_net)
        self.recog_net = Dis_net(**cfg.model.recog_net)
        self.generator = TopNGenerator(cfg.model.generator)

        self.end_MLP = End_MLP(cfg.model.end_MLP)
        self.react_score_MLP = End_MLP(cfg.model.react_score_MLP)

        if os.path.exists(cfg.dataset.vocab_loc):
            self.vocab = pickle.load(open(cfg.dataset.vocab_loc, 'rb'))
        else:
            self.vocab = pickle.load(open('.' + cfg.dataset.vocab_loc, 'rb'))
    
    def forward(self, batch_data, mode='train'):
        path, path_lens, target_graphs, target_graphs_len, target_reagents, target_reagent_lens, target_lens, end, fp = batch_data
        if not self.ablation:
            target_encoding_all, target_encoding = self.target_encoder(target_graphs, target_graphs_len, target_reagents, target_reagent_lens)
        else:
            target_encoding_all = torch.sum(fp, dim=1)
            target_encoding = [fp[i, :target_lens[i], :] for i in range(fp.shape[0])]

        context = self.path_encoder(path, path_lens)
        recog_mu, recog_logvar = self.recog_net(torch.cat([context, target_encoding_all], dim=-1))
        z_recog = sample_gaussian(recog_mu, recog_logvar)
        prior_mu, prior_logvar = self.prior_net(context)
        z_prior = sample_gaussian(prior_mu, prior_logvar)
        context_enc = context.clone()
        if mode == 'train':
            context = torch.cat((context, z_recog), axis=1)
        else:
            context = torch.cat((context, z_prior), axis=1)
        predicted_end = self.end_MLP(context)
        target_lens_org = copy.deepcopy(target_lens)
        results, results_mask, predicted_n = self.generator(context, target_lens)
        results_score = self.react_score_MLP(results.view(-1, results.shape[-1]))
        n_loss = nn.BCELoss()(results_score, results_mask.view(-1, 1))
        results = results * results_mask
        #n_loss = nn.MSELoss()(predicted_n, torch.tensor(target_lens_org).float().to(predicted_n.device).unsqueeze(-1))
        end_loss = nn.BCELoss()(predicted_end, end.unsqueeze(-1).float())
        result_collect = []
        target_collect = []
        for ind, decode_len in enumerate(target_lens):
            decode_len = min(decode_len, target_encoding[ind].shape[0])
            result_collect.append(results[ind, :decode_len,:])
            target_collect.append(target_encoding[ind][:decode_len,:])
        result_collect = torch.cat(result_collect, axis=0)
        target_collect = torch.cat(target_collect, axis=0)
        result_collect = F.normalize(result_collect, dim=-1)
        target_collect = F.normalize(target_collect, dim=-1)
        #print(result_collect.shape, target_collect.shape)
        #result_loss = self.contrastive_loss(result_collect, target_collect)
        accuracy = torch.tensor(0.0)
        sim_matrix = torch.matmul(result_collect, target_collect.transpose(0,1))
        for idx in range(result_collect.shape[0]):
            rank = torch.argsort(sim_matrix[idx,:], descending=True).tolist()
            accuracy += (rank.index(idx) / len(rank))
        accuracy = accuracy / result_collect.shape[0]
        result_loss = - torch.mean(torch.sum(result_collect * target_collect, dim=-1))
        kl_loss = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar).mean()
        return {'n_loss': n_loss, 'end_loss': end_loss, 'result_loss': result_loss, 'kl_loss': kl_loss,
                'accuracy': accuracy}


    @torch.no_grad()
    def generate_target_rep(self, smiles, role, device):
        if role == 'reactant':
            mol = Chem.MolFromSmiles(smiles)
            graph_item = preprocess_item(mol_to_graph(mol))
            graph_item.idx = 0
            graph_item = graph_collator([graph_item])
            graph_item['mask'] = torch.ones(graph_item['x'].size(0), graph_item['x'].size(1) + 1, dtype=torch.bool)
            for k, v in graph_item.items():
                if isinstance(v, torch.Tensor):
                    graph_item[k] = v.to(device)
            output = self.target_encoder(graph_item, [1], None, [0])
            return output[1]
        elif role == 'reagent':
            smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)
            reactant_vec = parser_selfies.smiles_to_vec(chem_utils.can_smiles(smiles),
                                                        self.vocab['symbol_to_idx'],
                                                        add_aux=True)
            utter_data = torch.tensor(reactant_vec, dtype=torch.long).unsqueeze(0).unsqueeze(0)
            utter_data = utter_data.to(device)
            output = self.target_encoder(None, [0], utter_data, [1])
            return output[1]

    @torch.no_grad()
    def generate_reactant_rep(self, paths, device):
        graph_items = []
        for path in paths:
            for s in path:
                s = Chem.MolFromSmiles(chem_utils.can_smiles(Chem.MolToSmiles(s)))
                graph_items.append(preprocess_item(mol_to_graph(s)))
        paths_len = [len(path) for path in paths]
        for idx, graph in enumerate(graph_items):
            graph.idx = idx
        graphs = graph_collator(graph_items)
        graphs['mask'] = torch.zeros(graphs['x'].size(0), graphs['x'].size(1) + 1, dtype=torch.bool)# +1 for virtual node
        for ind in range(graphs['x'].size(0)):
            graphs['mask'][ind, :graph_items[ind]['x'].shape[0] + 1] = True
        for k, v in graphs.items():
            if isinstance(v, torch.Tensor):
                graphs[k] = v.to(device)
        output = self.path_encoder(graphs, paths_len)
        return output

    @torch.no_grad()#TODO: make this parrallel
    def generate(self, path, device):
        path_rep = self.generate_reactant_rep(path, device)
        prior_mu, prior_logvar = self.prior_net(path_rep)
        z = sample_gaussian(prior_mu, prior_logvar)
        context = torch.cat([path_rep, z], dim=1)
        predicted_end = self.end_MLP(context)
        results, results_mask, n = self.generator(context)
        predicted_n = []
        for i in range(results.shape[0]):
            react_score = self.react_score_MLP(results[i, :, :].view(-1, self.cfg.model.generator.hidden_dim))
            try:
                p_n = max(torch.where(react_score > 0.5)[0])
                predicted_n.append(p_n.int().item())
            except:
                predicted_n.append(1)
        #remove pad from results
        results_clean = []
        for i in range(results.shape[0]):
            results_clean.append(results[i, :predicted_n[i]])
        return results_clean, predicted_end
        
    
    @torch.no_grad()
    def generate_condition(self, path, ref_target, ref_label, device):#optimize to maximize the label
        path_rep = self.generate_reactant_rep(path, device)
        prior_mu, prior_logvar = self.prior_net(path_rep)
        z = sample_gaussian(prior_mu, prior_logvar)
        context = torch.cat([path_rep, z], dim=1)
        predicted_end = self.end_MLP(context)
        results, results_mask, n = self.generator(context)
        predicted_n = []
        for i in range(results.shape[0]):
            react_score = self.react_score_MLP(results[i, :, :].view(-1, self.cfg.model.generator.hidden_dim))
            try:
                p_n = max(torch.where(react_score > 0.5)[0])
                predicted_n.append(p_n.int().item())
            except:
                predicted_n.append(1)
        #remove pad from results
        results_clean = []
        for i in range(results.shape[0]):
            results_clean.append(results[i, :predicted_n[i]])
        return results_clean, predicted_end


    def training_step(self, batch, batch_idx):
        result = self.forward(batch, mode='train')
        loss = result['n_loss'] + result['end_loss'] + result['result_loss'] * self.cfg.model.result_weight + result['kl_loss'] * self.cfg.model.kl_weight
        for k, v in result.items():
            if k.endswith('loss'):
                self.log(f'train_{k}', v,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.cfg.dataset.loader.batch_size)
        self.log('train_accuracy', result['accuracy'],
                on_epoch=True,
                prog_bar=True,
                batch_size=self.cfg.dataset.loader.batch_size)

        return loss
        
    
    
    def contrastive_loss(self, z1, z2):
        assert z1.shape == z2.shape
        bs = z1.shape[0]
        loss = info_nce.InfoNCE(temperature=0.2, negative_mode='paired')
        neg = []
        neg_keys = []
        for i in range(bs):
            neg_k = [j for j in range(bs) if j != i and ((z2[j]- z2[i])**2).sum()>1e-6]
            neg_keys.append(neg_k)
        neg_size = min([len(k) for k in neg_keys])
        for i in range(bs):
            neg_k = random.sample(neg_keys[i], neg_size)
            neg.append(z2[neg_k])
        neg = torch.stack(neg, dim=0)
        return loss(z1, z2, neg)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optim.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        scheduler = {'scheduler': scheduler, 'monitor': 'train_result_loss', 'interval': 'epoch', 'frequency': 1}

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
        return result
    
    def validation_step(self, batch, batch_idx):
        result = self.forward(batch, mode='val')
        loss = {k: v for k, v in result.items() if k.endswith('loss') or k == 'accuracy'}
        return loss

    

    def validation_epoch_end(self, result):
        metrics = self._compute_metrics(self._gather_result(result))
        for k, v in metrics.items():
            self.log(f'val_{k}', v, on_epoch=True, prog_bar=True, batch_size=self.cfg.dataset.loader.batch_size)
    
    def test_step(self, batch, batch_idx):
        result = self.forward(batch, mode='test')
        loss = {k: v for k, v in result.items() if k.endswith('loss') or k == 'accuracy'}
        return loss
    
    def test_epoch_end(self, result):
        metrics = self._compute_metrics(self._gather_result(result))
        for k, v in metrics.items():
            self.log(f'test_{k}', v, on_epoch=True, prog_bar=True, batch_size=self.cfg.dataset.loader.batch_size)

    
    
    def _gather_result(self, result):
        # collect steps
        result = {
            key: torch.cat([x[key] for x in result])
            if len(result[0][key].shape) > 0
            else torch.tensor([x[key] for x in result]).to(result[0][key])
            for key in result[0].keys()
        }
        #collect machines
        result = {
            key: torch.cat(list(self.all_gather(result[key])))
            for key in result.keys()
        }
        return result
        
