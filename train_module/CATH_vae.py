import copy
import logging
import math
import os
import pickle
import random
from typing import Any, Dict

import dataset.chem_utils_for_reactant as chem_utils
import dataset.parser_selfies as parser_selfies
import info_nce
import torch
from data_utils.dataset_graph import graph_collator
from data_utils.parser import mol_to_graph, preprocess_item
from model.Set_modules_vae import *
from pytorch_lightning import LightningModule
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CATH(LightningModule):
    '''
    Uni-RXN generative model(LightningModule)
    '''
    def __init__(self,cfg: Dict[str,Any], stage: str = 'pretrain'):
        """
        args:
            cfg: config file for model
            stage: pretrain or inferencing
        """
        super().__init__()
        self.cfg = cfg
        #self.save_hyperparameters()
        self.ablation = False
        self.epoch_num = 0

        self.path_encoder = Path_encoder(cfg.model)
        if stage == 'pretrain':
            #Only run when use pretraining weights (remember to cancel when inferencing)
            self.path_encoder.load_checkpoint(cfg.model.pretrained_path)
        self.target_encoder = Target_encoder(cfg.model)
        
        if stage == 'pretrain':
            #Only run when use pretraining weights (remember to cancel when inferencing)
            self.target_encoder.load_checkpoint(cfg.model.pretrained_path)
        self.prior_net = Dis_net(**cfg.model.prior_net)
        self.recog_net = Dis_net(**cfg.model.recog_net)
        self.generator = TopNGenerator(cfg.model.generator)

        self.end_MLP = End_MLP(cfg.model.end_MLP)
        self.react_score_MLP = End_MLP(cfg.model.react_score_MLP)

        self.vocab = pickle.load(open('./dataset/data/selfies_vocab.pkl', 'rb'))

        self.condition = cfg.model.condition

        #for conditional generation
        if self.condition:
            condition_net = nn.Sequential(nn.Linear(cfg.model.latent_dim, cfg.model.latent_dim), nn.BatchNorm1d(cfg.model.latent_dim), nn.ReLU(), nn.Linear(cfg.model.latent_dim, 1))
            self.condition_net = nn.ModuleList([copy.deepcopy(condition_net) for _ in range(8)])
    
    def forward(self, batch_data, mode='train'):
        path, path_lens, target_graphs, target_graphs_len, target_reagents, target_reagent_lens, target_lens, end,condition_label, contrast_mask = batch_data

        target_encoding_all, target_encoding = self.target_encoder(target_graphs, target_graphs_len, target_reagents, target_reagent_lens)
        target_encoding_all = target_encoding_all.detach()
        target_encoding = [d.detach() for d in target_encoding]


        context = self.path_encoder(path, path_lens)
        recog_mu, recog_logvar = self.recog_net(torch.cat([context, target_encoding_all], dim=-1))
        z_recog = sample_gaussian(recog_mu, recog_logvar)
        prior_mu, prior_logvar = self.prior_net(context)
        z_prior = sample_gaussian(prior_mu, prior_logvar)

        if mode == 'train':
            context = torch.cat((context, z_recog), axis=1)
        else:
            context = torch.cat((context, z_prior), axis=1)
        predicted_end = self.end_MLP(context)
        target_lens_org = copy.deepcopy(target_lens)
        results, results_mask, predicted_n = self.generator(context, target_lens)
        condition_predict = [self.condition_net[i](context) for i in range(8)]
        condition_label = list(condition_label.values())
        condition_loss = [nn.MSELoss()(condition_predict[i], condition_label[i].unsqueeze(-1).float().to(context.device)) for i in range(8)]
        condition_loss_dict = {'condition_'+str(i)+'_loss':condition_loss[i] for i in range(8)}
        condition_loss = sum(condition_loss)
        results_score = self.react_score_MLP(results.view(-1, results.shape[-1]))
        n_loss = nn.BCELoss()(results_score, results_mask.view(-1, 1))
        results = results * results_mask
        end_loss = nn.BCELoss()(predicted_end, end.unsqueeze(-1).float())
        result_collect = []
        target_collect = []
        for ind, decode_len in enumerate(target_lens):
            decode_len = min(decode_len, target_encoding[ind].shape[0])
            result_collect.append(results[ind, :decode_len,:])
            target_collect.append(target_encoding[ind][:decode_len,:])
            result_collect[-1], target_collect[-1] = self.opt_permute(result_collect[-1], target_collect[-1])
        result_collect = torch.cat(result_collect, axis=0)
        target_collect = torch.cat(target_collect, axis=0)
        result_collect = F.normalize(result_collect, dim=-1)
        target_collect = F.normalize(target_collect, dim=-1)

        result_loss = self.contrastive_loss(result_collect, target_collect)#, mask=contrast_mask)
        accuracy = torch.tensor(0.0)
        sim_matrix = torch.matmul(result_collect, target_collect.transpose(0,1))
        for idx in range(result_collect.shape[0]):
            rank = torch.argsort(sim_matrix[idx,:], descending=True).tolist()
            accuracy += (rank.index(idx) / len(rank))
        accuracy = accuracy / result_collect.shape[0]

        kl_loss = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar).mean()


        return {'n_loss': n_loss, 'end_loss': end_loss, 'result_loss': result_loss, 'kl_loss': kl_loss,
                'accuracy': accuracy, 'condition_loss': condition_loss, **condition_loss_dict}
    
    def opt_permute(self, x1, x2):
        distances = torch.cdist(x1, x2)
        row_ind, col_ind = linear_sum_assignment(distances.cpu().detach().numpy())
        return x1[row_ind], x2[col_ind]


    @torch.no_grad()
    def generate_target_rep(self, smiles, role, device):
        """
        Generate target representation for a given smiles, smiles can be reactant or reagent
        :param smiles: smiles string
        :param role: 'reactant' or 'reagent'
        :param device: torch device (cpu or cuda)
        """
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
            for path_id in range(0, len(path), 2):
                s = path[path_id]
                if s is not None:
                    s = Chem.MolFromSmiles(Chem.MolToSmiles(s))
                graph_items.append(preprocess_item(mol_to_graph(s)))
        paths_len = [len(path) //2 + 1 for path in paths]
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

    @torch.no_grad()
    def generate(self, path, device, denovo=False):
        path_rep = self.generate_reactant_rep(path, device)
        path_first = [p[0] is None for p in path]
        prior_mu, prior_logvar = self.prior_net(path_rep)
        z = sample_gaussian(prior_mu, prior_logvar)
        context = torch.cat([path_rep, z], dim=1)
        predicted_end = self.end_MLP(context)
        if denovo:
            predicted_mw = self.condition_net[2](context)
        results, results_mask, _ = self.generator(context)
        predicted_n = []
        for i in range(results.shape[0]):
            if path_first[i]:
                predicted_n.append(1)
                continue
            react_score = self.react_score_MLP(results[i, :, :].view(-1, self.cfg.model.generator.hidden_dim))
            try:
                p_n = max(torch.where(react_score > 0.5)[0])
                predicted_n.append(p_n.int().item() + 1)
            except:
                predicted_n.append(1)
        #remove pad from results
        results_clean = []
        for i in range(results.shape[0]):
            results_clean.append(results[i, :predicted_n[i]])
        if denovo:
            return results_clean, predicted_mw
        else:
            return results_clean, predicted_end
    
    @torch.no_grad()
    def generate_condition(self, path, device, condition_num, condition_direction, iteration=1):
        with torch.no_grad():
            path_first = [len(p) == 1 for p in path]
            path_rep = self.generate_reactant_rep(path, device)
            prior_mu, prior_logvar = self.prior_net(path_rep)
            z = sample_gaussian(prior_mu, prior_logvar)
        for _ in range(iteration):
            context = torch.cat([path_rep, z], dim=1)
            #predict the condition and backprop the gradient to z
            condition = self.condition_net[condition_num](context)
            z_grad = torch.autograd.grad(condition.sum(), z, retain_graph=True)[0]
            z = z + condition_direction * z_grad

        context = torch.cat([path_rep, z], dim=1)
        with torch.no_grad():
            predicted_end = self.end_MLP(context)
            results, results_mask, _ = self.generator(context)
            predicted_n = []
        for i in range(results.shape[0]):
            if path_first[i]:
                predicted_n.append(1)
                continue
            with torch.no_grad():
                react_score = self.react_score_MLP(results[i, :, :].view(-1, self.cfg.model.generator.hidden_dim))
            try:
                p_n = max(torch.where(react_score > 0.5)[0])
                predicted_n.append(p_n.int().item() + 1)
            except:
                predicted_n.append(1)
        #remove pad from results
        results_clean = []
        for i in range(results.shape[0]):
            results_clean.append(results[i, :predicted_n[i]])
        return results_clean, predicted_end

    


    def training_step(self, batch, batch_idx):
        result = self.forward(batch, mode='train')
        loss = result['n_loss'] + result['end_loss']\
               + result['result_loss'] * self.cfg.model.result_weight \
               + result['kl_loss'] * self.cfg.model.kl_weight * max(self.epoch_num/100, 1)\
               + result['condition_loss'] * self.cfg.model.condition_weight
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
        
    
    
    def contrastive_loss(self, z1, z2, mask):
        assert z1.shape == z2.shape
        assert mask.shape[0] == z1.shape[0]
        bs = z1.shape[0]
        if mask is not None:
            mask = ~mask
            loss = info_nce.InfoNCE(negative_mode='paired')
            neg_size = min(mask.sum(dim=1))
            neg = []
            for i in range(mask.shape[0]):
                neg_sample = mask[i, :].nonzero().squeeze()
                neg_sample = random.sample(neg_sample.tolist(), neg_size)
                neg.append(z2[neg_sample])
            neg = torch.stack(neg, dim=0)
            return loss(z1, z2, neg)
        else:
            loss = info_nce.InfoNCE()
            return loss(z1, z2)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optim.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        scheduler = {'scheduler': scheduler, 'monitor': 'train_result_loss', 'interval': 'epoch', 'frequency': 1}

        return [optimizer], [scheduler]
    

    def _compute_metrics(self, result):
        return {k: l.mean() for k, l in result.items()}
    
    def validation_step(self, batch, batch_idx):
        result = self.forward(batch, mode='val')
        loss = {k: v for k, v in result.items() if k.endswith('loss') or k == 'accuracy'}
        return loss

    

    def validation_epoch_end(self, result):
        metrics = self._compute_metrics(self._gather_result(result))
        for k, v in metrics.items():
            self.log(f'val_{k}', v, on_epoch=True, prog_bar=True, batch_size=self.cfg.dataset.loader.batch_size)
        self.epoch_num += 1
    
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
        
