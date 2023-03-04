import time

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.autograd import Variable


class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."

    def __init__(self, generator, criterion, devices, opt=None, reg_weight=100, chunk_size=6):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.reg_weight = reg_weight

    def __call__(self, out, targets, memory, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        distribute_loss = compute_mmd(memory, self.reg_weight/ (out.shape[0] * (out.shape[0] - 1)))
        gen = nn.parallel.parallel_apply(generator, out_scatter)
        y = [(g.contiguous(), t.contiguous()) for g, t in zip(gen, targets)]
        gen_loss = nn.parallel.parallel_apply(self.criterion, y)

        # Sum and normalize loss
        gen_l = nn.parallel.gather(gen_loss, target_device=self.devices[0])
        gen_l = gen_l.sum() / normalize
        #print(f'label loss: {gen_l} distribute_loss: {distribute_loss}')
        all_loss = gen_l + distribute_loss

        # Backprop loss 
        if self.opt is not None:
            all_loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return gen_l * normalize, distribute_loss, all_loss


# Partly modified from https://github.com/AntixK/PyTorch-VAE/blob/master/models/wae_mmd.py
def compute_rbf(x1, x2, eps: float = 1e-7):
    """
    Computes the RBF Kernel between x1 and x2.
    :param x1: (Tensor)
    :param x2: (Tensor)
    :param eps: (Float)
    :return:
    """
    z_dim = x2.size(-1)
    sigma = 2.0 * z_dim * 2

    result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
    return result


def compute_inv_mult_quad(x1, x2, eps: float = 1e-7):
    """
    Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
    given by
            k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
    :param x1: (Tensor)
    :param x2: (Tensor)
    :param eps: (Float)
    :return:
    """
    z_dim = x2.size(-1)
    C = 2 * z_dim * 2
    kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

    # Exclude diagonal elements
    b = kernel.shape[0]
    result = kernel.sum() - torch.diag(kernel).sum()
    return result


def compute_kernel(x1, x2, kernel_type="imq"):
    # Convert the tensors into row and column vectors

    x1 = x1.unsqueeze(-2)  # Make it into a column tensor
    x2 = x2.unsqueeze(-3)  # Make it into a row tensor

    if kernel_type == "rbf":
        result = compute_rbf(x1, x2)
    elif kernel_type == "imq":
        result = compute_inv_mult_quad(x1, x2)
    else:
        raise ValueError("Undefined kernel type.")

    return result


def compute_mmd(z, reg_weight: float):
    # Sample from prior (Gaussian) distribution
    prior_z = torch.randn_like(z)

    prior_z__kernel = compute_kernel(prior_z, prior_z)
    z__kernel = compute_kernel(z, z)
    priorz_z__kernel = compute_kernel(prior_z, z)

    mmd = (
        reg_weight * prior_z__kernel.mean()
        + reg_weight * z__kernel.mean()
        - 2 * reg_weight * priorz_z__kernel.mean()
    )
    return mmd


class WAE_loss(nn.Module):
    def __init__(self, pad_idx, smooth=0.0):
        super(WAE_loss, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduction='mean')
        self.padding_idx = pad_idx
        self.confidence = 1.0 - smooth
        self.smooth = smooth

    def forward(self, x, target):
        true_dist = torch.zeros(x.shape).to(target.device)
        size = x.size(1)
        true_dist.fill_(self.smooth / (size - 2))
        true_dist.scatter_(2, target.unsqueeze(1), self.confidence)
        true_dist[:, :, self.padding_idx] = 0
        mini_mask = (target.data == self.padding_idx)
        mask = torch.stack([mini_mask for i in range(x.shape[-1])], dim=-1)
        true_dist = torch.where(mask==True, torch.zeros_like(true_dist), true_dist)
        generate_loss = self.criterion(x, Variable(true_dist, requires_grad=False))
        return generate_loss


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


def get_std_opt(model):
    return NoamOpt(
        model.Decoder_d,
        2,
        4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
    )

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, graph, src, trg=None, pad=0):
        self.graph = graph
        self.src = src[:, :-1]
        self.src_mask = (src[:, :-1] != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
        
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def rebatch(pad_idx, batch, device):
    "Fix order in torchtext to match ours"
    #src, trg = batch[0].transpose(0, 1), batch[1].transpose(0, 1)
    graph = batch[0]
    for i in range(3):
        graph[i] = graph[i].cuda()
    src, trg = batch[1], batch[2]
    src = src.cuda()
    trg = trg.cuda()
    return Batch(graph, src, trg, pad_idx)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    
def run_epoch(data_iter, model, loss_compute, device):
    start = time.time()
    total_tokens = 0
    total_all_loss, total_label_loss, total_distribute_loss = 0, 0, 0
    tokens = 0
    batch_steps = len(data_iter)
    print(f'got {batch_steps} batches to train in this epoch')
    epoch_start_time = time.time()
    for i, batch in enumerate(data_iter):
        if i < batch_steps - 2:
            batch_m = rebatch(0, batch, device)
            out, memory = model.forward(batch_m.graph[0], batch_m.graph[1], batch_m.graph[2],  
                                batch_m.src, batch_m.trg, 
                                batch_m.src_mask, batch_m.trg_mask)
            label_loss, distribute_loss, all_loss = loss_compute(out, batch_m.trg_y, memory, batch_m.ntokens)
            total_all_loss += all_loss
            total_label_loss += label_loss
            total_distribute_loss += distribute_loss
            total_tokens += batch_m.ntokens
            tokens += batch_m.ntokens
            if i % 3 == 1:
                elapsed = time.time() - start
                print(f"Epoch Step: {i} Label Loss: {total_label_loss / batch_m.ntokens} Distribute Loss: {total_distribute_loss / batch_steps} Tokens per Sec: {tokens / elapsed} time needed: {(time.time() - epoch_start_time) / i * (batch_steps - i) / 3600}" 
                        )
                start = time.time()
                tokens = 0
    return total_all_loss / batch_steps
