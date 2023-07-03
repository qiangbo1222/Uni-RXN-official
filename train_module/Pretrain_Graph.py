import imp
import logging
import math
import pickle
from typing import Any, Dict

import dataset.chem_utils_for_reactant as chem_utils
import dataset.parser_selfies as parser_selfies
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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Pretrain_Graph(LightningModule):
    def __init__(self,cfg: Dict[str,Any], stage='pretrain'):
        super().__init__()
        self.cfg = cfg
        self.state = stage
        #self.save_hyperparameters()
        self.react_graph_model = graph_encoder(cfg.model.graph_encoder)
        self.prod_graph_model = graph_encoder(cfg.model.graph_encoder)

        pos_embed = Pos_embeddings(**cfg.model.pos_embed)
        self.reagent_model = utter_encoder(**cfg.model.utter_encoder, embedder=pos_embed)
        self.reagent_encoder = agg_encoder(**cfg.model.reagent_encoder)

        self.center = cfg.model.center
        if self.center:
            self.center_predictor = nn.Sequential(MLP_head(**cfg.model.center_predictor), 
                                                    nn.Sigmoid())
            self.react_graph_model_cross = graph_encoder(cfg.model.graph_encoder_cross, cross=True)
            
        self.input_head = MLP_head(**cfg.model.graph_head)
        self.reactants_head = MLP_head(**cfg.model.graph_head)
        self.graph_head = MLP_head(**cfg.model.graph_head)
        self.reagent_head = MLP_head(**cfg.model.reagent_head)
        self.finetune_head = MLP_head(**cfg.model.finetune_head)

        self.loss_lambda = cfg.model.loss_lambda
        
        self.vocab = pickle.load(open("dataset/data/selfies_vocab.pkl", 'rb'))

    def forward(self, batch):
        if self.state == 'pretrain':
            return self.pretrain(batch)
        elif self.state == 'finetune':
            return self.finetune(batch)
    
    def pretrain(self, batch):
        graph_input_encoding = self.react_graph_model(batch['graph_input'])
        graph_reactants_encoding = self.react_graph_model(batch['graph_reactants'])

        bs, reactant_num_max, reactant_len = batch['utter'].shape
        reagent_num = batch['reagent_num']

        utter = batch['utter'].reshape(bs*reactant_num_max, -1)
        utter_encoding = self.reagent_model(utter)
        utter_encoding = utter_encoding.reshape(bs, reactant_num_max, reactant_len, -1)
        utter_encoding = utter_encoding[:, :, 0, :]
        reagent_encoding = self.reagent_encoder(utter_encoding, reagent_num)

        graph_prod_encoding = self.prod_graph_model(batch['graph_prod'])[:, 0, :]


        graph_input_pred = self.input_head(graph_input_encoding[:, 0, :])
        graph_reactants_pred = self.reactants_head(graph_reactants_encoding[:, 0, :])
        graph_pred = [self.graph_head(graph_input_encoding[:, 0, :]),
                      self.graph_head(graph_reactants_encoding[:, 0, :]),
                      self.graph_head(graph_prod_encoding)]
        reagent_pred = self.reagent_head(reagent_encoding)

        
        graph_loss = self.contrastive_loss(graph_pred[0] + graph_pred[1], graph_pred[2])
        graph_loss += self.contrastive_loss(graph_pred[2], graph_pred[0] + graph_pred[1])
        reaction_loss = self.contrastive_loss(graph_input_pred, graph_reactants_pred + reagent_pred)
        reaction_loss += self.contrastive_loss(graph_reactants_pred + reagent_pred, graph_input_pred)#permute the pairing task

        accuracy = torch.tensor(0.0)
        sim_matrix = torch.matmul(graph_input_pred, (graph_reactants_pred + reagent_pred).transpose(-2, -1))
        for idx in range(bs):
            if torch.argmax(sim_matrix[idx]) == idx:
                accuracy += 1
        accuracy /= bs


        cache = graph_input_encoding[:, 0, :]
        graph_input_encoding += (graph_reactants_encoding[:, 0, :].unsqueeze(1).repeat(1, graph_input_encoding.shape[1], 1) + \
                                    reagent_encoding.unsqueeze(1).repeat(1, graph_input_encoding.shape[1], 1))
        graph_reactants_encoding += (cache.unsqueeze(1).repeat(1, graph_reactants_encoding.shape[1], 1)+ \
                                    reagent_encoding.unsqueeze(1).repeat(1, graph_reactants_encoding.shape[1], 1))


        #predict the reaction center
        if self.center:
            #cross attention
            graph_input_encoding = self.react_graph_model_cross(batch['graph_input'], graph_input_encoding)
            graph_reactants_encoding  = self.react_graph_model_cross(batch['graph_reactants'], graph_reactants_encoding)
            input_center_pred = self.center_predictor(graph_input_encoding[:, 1:, :])
            reactants_center_pred = self.center_predictor(graph_reactants_encoding[:, 1:, :])
            input_center, reactants_center = batch['change_idx'][0].unsqueeze(-1), batch['change_idx'][1].unsqueeze(-1)
            input_center_loss = nn.BCELoss()(input_center_pred, input_center.float())
            reactants_center_loss = nn.BCELoss()(reactants_center_pred, reactants_center.float())
            center_loss = input_center_loss + reactants_center_loss

            center_accuracy = (torch.round(input_center_pred) == input_center).sum().float() / (input_center.shape[0] * input_center.shape[1])
            center_accuracy += (torch.round(reactants_center_pred) == reactants_center).sum().float() / (reactants_center.shape[0] * reactants_center.shape[1])
            center_accuracy /= 2

            return {'graph_loss': graph_loss, 'reaction_loss': reaction_loss,
                    'center_loss': center_loss, 'center_accuracy': center_accuracy,
                    'loss': graph_loss * self.loss_lambda + reaction_loss + center_loss, 
                    'accuracy': accuracy}
        else:
            return {'graph_loss': graph_loss, 'reaction_loss': reaction_loss,
                    'loss': graph_loss * self.loss_lambda + reaction_loss, 
                    'accuracy': accuracy}
    
    def contrastive_loss(self, z1, z2):
        bs = z1.shape[0]
        loss = info_nce.InfoNCE(negative_mode='paired')
        neg = []
        for i in range(bs):
            neg_keys = [j for j in range(bs) if j != i]
            neg.append(z2[neg_keys])
        neg = torch.stack(neg, dim=0)
        return loss(z1, z2, neg)
    
    def finetune(self, batch):
        graph_reactants_encoding = self.react_graph_model(batch['graph_reactants'])

        classify_log = self.finetune_head(graph_reactants_encoding[:, 0, :])
        classify_loss = nn.CrossEntropyLoss()(classify_log, batch['labels'])
        accuracy = (torch.argmax(classify_log, dim=-1) == batch['labels']).sum().float() / batch['labels'].shape[0]
        return {'classify_loss': classify_loss, 'accuracy': accuracy, 'loss': classify_loss}
    
    @torch.no_grad()
    def generate_reactant_fp(self, mols, device):
        if not isinstance(mols, list):
            mols = [mols]
        graph_item = [preprocess_item(mol_to_graph(mol)) for mol in mols]
        for idx, item in enumerate(graph_item):
            item.idx = idx
        graph_item_assem = graph_collator(graph_item)
        graph_item_assem['mask'] = torch.zeros(graph_item_assem['x'].size(0), graph_item_assem['x'].size(1) + 1, dtype=torch.bool)
        for ind in range(graph_item_assem['x'].size(0)):
            graph_item_assem['mask'][ind, :graph_item[ind]['x'].shape[0] + 1] = True

        for k, v in graph_item_assem.items():
            if isinstance(v, torch.Tensor):
                graph_item_assem[k] = v.to(device)
        return self.react_graph_model(graph_item_assem), graph_item_assem
    
    @torch.no_grad()
    def generate_product_fp(self, mol, device):
        graph_item = preprocess_item(mol_to_graph(mol))
        graph_item.idx = 0
        graph_item = graph_collator([graph_item])
        graph_item['mask'] = torch.zeros(graph_item['x'].size(0), graph_item['x'].size(1) + 1, dtype=torch.bool)
        for k, v in graph_item.items():
            if isinstance(v, torch.Tensor):
                graph_item[k] = v.to(device)
        return self.prod_graph_model(graph_item), graph_item


    @torch.no_grad()
    def generate_reagent_fp(self, smi_list, device):
        smi_list = [chem_utils.can_smiles(smi) for smi in smi_list if len(smi)> 0 and smi is not None]
        reactant_vec = [parser_selfies.smiles_to_vec(smi, 
                                                       self.vocab['symbol_to_idx'], 
                                                       add_aux=True) for smi in smi_list if len(smi)> 0 and smi is not None]
        reactant_num = len(reactant_vec)
        if reactant_num == 0:
            return torch.zeros(1, self.cfg.model.utter_encoder.hidden_size).to(device)
        else:
            max_len_reactants = max([len(r) for r in reactant_vec])
        utter_data = torch.zeros([reactant_num,  max_len_reactants], dtype=torch.long)
        for i, vec in enumerate(reactant_vec):
            utter_data[i, :len(vec)] = torch.tensor(vec)
        utter_data = utter_data.to(device)
        utter_encoding = self.reagent_model(utter_data)
        utter_encoding = utter_encoding.reshape(1, reactant_num, max_len_reactants, -1)
        utter_encoding = utter_encoding[:, :, 0, :]
        reagent_encoding = self.reagent_encoder(utter_encoding, [reactant_num])
        return reagent_encoding

        
    @torch.no_grad()
    def generate_utter_fp(self, mol, device):
        smiles = Chem.MolToSmiles(mol)
        reactant_vec = parser_selfies.smiles_to_vec(chem_utils.can_smiles(smiles),
                                                    self.vocab['symbol_to_idx'],
                                                    add_aux=True)
        utter_data = torch.tensor(reactant_vec, dtype=torch.long).unsqueeze(0)
        utter_data = utter_data.to(device)
        utter_encoding = self.reagent_model(utter_data)
        return utter_encoding[0, 0, :]
    
    @torch.no_grad()
    def generate_reaction_fp(self, reaction_rxn, device):
        input_mol = Chem.MolFromSmiles(reaction_rxn[0][0][0])
        reactant_mol = Chem.MolFromSmiles('.'.join([reaction_rxn[0][1][i][0] for i in range(len(reaction_rxn[0][1]))]))
        reagent_list = reaction_rxn[1].split('.')
        input_fp, input_item = self.generate_reactant_fp(input_mol, device)
        reactant_fp, reactant_item = self.generate_reactant_fp(reactant_mol, device)
        reagent_fp = self.generate_reagent_fp(reagent_list, device)

        cache = input_fp[:, 0, :]
        input_fp += (reactant_fp[:, 0, :].unsqueeze(1).repeat(1, input_fp.shape[1], 1) + \
                                    reagent_fp.unsqueeze(1).repeat(1, input_fp.shape[1], 1))
        reactant_fp += (cache.unsqueeze(1).repeat(1, reactant_fp.shape[1], 1)+ \
                                    reagent_fp.unsqueeze(1).repeat(1, reactant_fp.shape[1], 1))

        #cross attention
        graph_input_encoding = self.react_graph_model_cross(input_item, input_fp)
        graph_reactants_encoding  = self.react_graph_model_cross(reactant_item, reactant_fp)
        
        return torch.cat([torch.sum(graph_input_encoding[:, 1:, :], dim=1), 
                          torch.sum(graph_reactants_encoding[:, 1:, :], dim=1),
                          reagent_fp], dim=1)
    
    @torch.no_grad()
    def generate_reaction_fp_mix(self, reaction_rxn, device, no_reagent=False):
        input_mol = reaction_rxn[0][0][0]
        if len(reaction_rxn[0][1]) > 0:
            reactant_mol = [input_mol] +  [reaction_rxn[0][1][i][0] for i in range(len(reaction_rxn[0][1]))]
            reactant_mol = [Chem.MolFromSmiles(m) for m in reactant_mol]
        else:
            reactant_mol = [Chem.MolFromSmiles(input_mol)]
        reactant_fp, reactant_item = self.generate_reactant_fp(reactant_mol, device)

        if not no_reagent:
            if isinstance(reaction_rxn[1], list):
                reagent_list = reaction_rxn[1]
            else:
                reagent_list = reaction_rxn[1].split('.')
            #product_fp, product_item = self.generate_product_fp(reaction_rxn[2], device)
            reagent_fp = self.generate_reagent_fp(reagent_list, device)

            return torch.cat([torch.sum(reactant_fp[:, 0, :], dim=0, keepdim=True), reagent_fp], dim=1)
        else:
            return torch.sum(reactant_fp[:, 0, :], dim=0, keepdim=True)
        
    

    
    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        for k, l in loss.items():
            self.log('train_' + k, l, on_epoch=True, prog_bar=True)
        return loss

    
    def configure_optimizers(self):
        if self.state == 'finetune':
            tune_param = list(self.finetune_head.parameters())
            base_param = list(self.react_graph_model.parameters())
            optimizer = torch.optim.Adam([{'params': tune_param, 'lr': self.cfg.optim.lr * 10},
                                            {'params': base_param, 'lr': self.cfg.optim.lr}], lr=self.cfg.optim.lr * 0.1)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optim.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.scheduler.step_size, gamma=self.cfg.scheduler.gamma)
        scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

        return [optimizer], [scheduler]
    
    def _compute_metrics(self, result):
        return {k: l.mean() for k, l in result.items()}
    
    def validation_step(self, batch, batch_idx):
        result = self.forward(batch)
        return result
    

    def validation_epoch_end(self, result):
        loss = self._compute_metrics(self._gather_result(result))
        for k, l in loss.items():
            self.log('valid_' + k, l, on_epoch=True, prog_bar=True)

    
    
    def _gather_result(self, result):
        # collect steps
        result = {
            key: torch.cat([x[key] for x in result])
            if len(result[0][key].shape) > 0
            else torch.tensor([x[key] for x in result]).to(result[0][key])
            for key in result[0].keys()
        }
        # collect machines
        result = {
            key: torch.cat(list(self.all_gather(result[key])))
            for key in result.keys()
        }
        return result
