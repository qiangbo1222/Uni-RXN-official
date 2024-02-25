#Partly borrowed from https://github.com/cvignac/Top-N/blob/85c7287780e15b9fb4eab419d9732e45a9bb67e8/set_generators.py
import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.net1 = nn.Sequential(nn.Linear(input_size, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, output_size))
        
    def forward(self, x):
        x = self.net1(x)
        out = self.net2(x)
        return out


def round_n(predicted_n, max_n):
    if predicted_n.dtype != torch.int64:
        predicted_n = torch.round(predicted_n)
    n = torch.clamp(predicted_n, min=1, max=max_n)
    return n


class SetGenerator(nn.Module):
    def __init__(self, cfg):
        """ Base class for a set generator. During training, the number of points n is assumed to be given.
            At generation time, if learn_from_latent, a value is predicted.
            Otherwise, a value is sampled from the train distribution
            n_distribution: dict. Each key is an integer, and the value the frequency at which sets of this
            size appear in the training set. """
        super().__init__()
        self.latent_channels = cfg.latent_dim
        self.set_channels = cfg.set_channels


        self.max_n = cfg.max_n
        self.mlp1 = MLP(self.latent_channels, cfg.hidden_dim, 1)


    def forward(self, latent: Tensor, n: int = None):
        """ A set generator returns a latent set with n nodes and set_channels features.
        Input: latent (Tensor of size batch x latent_channels)
        Returns: x (Tensor of size batch x n x set_channels).
                 n: int
        """
        predicted_n = self.mlp1(latent).squeeze(1)
        if n is None:
            n = self.generate_n(latent)

        return n, predicted_n

    def generate_n(self, z: Tensor = None):
        n = self.mlp1(z)
        return round_n(n, self.max_n)



class TopNGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.max_n = cfg.max_n
        self.set_channels = cfg.set_channels
        self.cosine_channels = cfg.cosine_channels
        self.points = nn.Parameter(torch.randn(cfg.point_num, cfg.set_channels).float())

        angles = torch.randn(cfg.point_num, cfg.cosine_channels).float()
        angles = angles / (torch.norm(angles, dim=1)[:, None] + 1e-5)
        self.angles_params = nn.Parameter(angles)

        self.angle_mlp = MLP(cfg.latent_dim, cfg.hidden_dim, self.cosine_channels)


        self.lin1 = nn.Linear(1, cfg.set_channels)
        self.lin2 = nn.Linear(1, cfg.set_channels)

        self.out_lin = nn.Sequential(nn.Linear(cfg.set_channels + cfg.latent_dim, cfg.set_channels + cfg.latent_dim),
                                     nn.ReLU(),
                                     nn.Linear(cfg.set_channels + cfg.latent_dim, cfg.set_channels),
                                     nn.ReLU(),
                                     nn.Linear(cfg.set_channels, cfg.set_channels))

    def forward(self, latent: Tensor, n: int = None):
        """ latent: batch_size x d
            self.points: max_points x d"""
        batch_size = latent.shape[0]
        inference = (n is None)
        #n, predicted_n = super().forward(latent, n)
        if inference:
            predicted_n = torch.zeros(batch_size, dtype=torch.int64).to(latent.device) + self.max_n
            n = round_n(predicted_n, self.max_n)
        angles = self.angle_mlp(latent)
        angles = angles / (torch.norm(angles, dim=1)[:, None] + 1e-5)

        cosine = (self.angles_params[None, ...] @ angles[:, :, None]).squeeze(dim=2)
        cosine = torch.softmax(cosine, dim=1)
        
        srted, indices = torch.topk(cosine, self.max_n, dim=1, largest=True, sorted=True)  # bs x n

        indices = indices[:, :, None].expand(-1, -1, self.points.shape[-1])  # bs, n, set_c
        batched_points = self.points[None, :].expand(batch_size, -1, -1)  # bs, n_max, set_c

        selected_points = torch.gather(batched_points, dim=1, index=indices)

        alpha = self.lin1(selected_points.shape[1] * srted[:, :, None])
        beta = self.lin2(selected_points.shape[1] * srted[:, :, None])
        modulated = alpha * selected_points + beta
        if not isinstance(n, list):
            n = n.tolist()
            n = [int(num) for num in n]
        modulated_list = [modulated[i, :n[i], :] for i in range(batch_size)]
        modulated_pad = torch.zeros([batch_size, self.max_n, self.set_channels]).to(latent.device)
        modulated_mask = torch.zeros([batch_size, self.max_n, 1]).to(latent.device)
        for i, m in enumerate(modulated_list):
            modulated_pad[i, :m.shape[0], :] = m
            modulated_mask[i, :m.shape[0], 0] = 1
        modulated = self.out_lin(torch.cat([modulated, latent.unsqueeze(1).repeat(1, self.max_n, 1)], dim=-1))
        return modulated, modulated_mask, n
    
