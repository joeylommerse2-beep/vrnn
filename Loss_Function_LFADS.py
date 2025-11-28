# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 10:06:39 2025

@author: joeyl
"""
import torch 

def poisson_loss(rates, x):
    return (rates - x * torch.log(rates + 1e-8)).sum(dim=(1,2)).mean()

def factor_orthogonality_loss(factors, eps=1e-8):
    """
    factors: (batch, T, F)
    """
    B, T, F = factors.shape
    F_flat = factors.reshape(B * T, F)          # (N, F)

    # Center and normalize each factor dimension
    F_flat = F_flat - F_flat.mean(dim=0, keepdim=True)
    F_flat = F_flat / (F_flat.std(dim=0, keepdim=True) + eps)

    # Correlation / Gram matrix between factor dimensions
    G = F_flat.T @ F_flat / (B * T)             # (F, F)

    # Zero the diagonal (self-correlation is fine)
    off_diag = G - torch.diag(torch.diag(G))

    # Penalize squared off-diagonal entries
    return (off_diag ** 2).mean()

def lfads_loss(rates, x, kl_ic, kl_ctrl, kl_weight, rec_weight, factors=None,
               ortho_weight=0.0):
    rec = poisson_loss(rates, x)
    total = rec_weight * rec + kl_weight * (kl_ic + kl_ctrl)
    if factors is not None and ortho_weight > 0:
        ortho = factor_orthogonality_loss(factors)
        total = total + ortho_weight * ortho

    return total, rec
