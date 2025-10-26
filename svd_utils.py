# -*- coding: utf-8 -*-
import torch


def compute_SVD(states, svd_mode, rank):
    n = 2
    if svd_mode == 'ht_d':
        if 'mps' in str(states.device):
            U_k, S_k, Vh_k = torch.svd_lowrank(states.view(states.shape[0], -1, states.shape[-1]).float().cpu(), q=rank, niter=n)
        else:
            U_k, S_k, Vh_k = torch.svd_lowrank(states.view(states.shape[0], -1, states.shape[-1]).float(), q=rank, niter=n)
    elif svd_mode == 't_hd':
        rank = min(rank, states.transpose(-2, -3).flatten(start_dim=-2).shape[-2])
        if 'mps' in str(states.device):
            U_k, S_k, Vh_k = torch.svd_lowrank(states.transpose(-2, -3).flatten(start_dim=-2).float().cpu(), q=rank, niter=n)
        else:
            U_k, S_k, Vh_k = torch.svd_lowrank(states.transpose(-2, -3).flatten(start_dim=-2).float(), q=rank, niter=n)
    else:
        rank = min([rank, states.shape[-1], states.shape[-2]])
        if 'mps' in str(states.device):
            U_k, S_k, Vh_k = torch.svd_lowrank(states.float().cpu(), q=rank, niter=n)
        else:
            U_k, S_k, Vh_k = torch.svd_lowrank(states.float(), q=rank, niter=n)
    return [U_k.half().to(states.device), 
            S_k.half().to(states.device), 
            Vh_k.transpose(-1, -2).half().to(states.device)]