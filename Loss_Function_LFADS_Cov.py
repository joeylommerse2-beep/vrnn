# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 10:06:39 2025

@author: joeyl
"""
import torch 

def poisson_loss(rates, x):
    return (rates - x * torch.log(rates + 1e-8)).sum(dim=(1,2)).mean()

def factor_orthogonality_loss(factors, eps=1e-8):
    F = factors.reshape(-1, F)
    F = F - F.mean(dim=0, keepdim=True)
    C = (F.T @ F) / F.shape[0]
    cov_penalty = ((C - torch.eye(F, device=C.device)) ** 2).mean()

    return cov_penalty

def lfads_loss(rates, x, kl_ic, kl_ctrl, kl_weight, rec_weight, lambda_cov=1e-3,
               factors):
    rec = poisson_loss(rates, x)
    cov_penalty = factor_orthogonality_loss(factors)
    total = rec_weight * rec + kl_weight * (kl_ic + kl_ctrl) + lambda_cov * cov_penalty
    return total, rec
