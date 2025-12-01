# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 10:06:39 2025

@author: joeyl
"""
import torch 

def poisson_loss(rates, x):
    return (rates - x * torch.log(rates + 1e-8)).sum(dim=(1,2)).mean()

def latent_mse_loss(factors, latents_true):
    """
    Supervised latent alignment for synthetic data:
    directly penalize MSE between factors and true latents.
    factors:      (B, T, F)
    latents_true: (B, T, K)  # here F == K == 3
    """
    return torch.mean((factors - latents_true) ** 2)

def lfads_loss(
    rates,
    x,
    kl_ic,
    kl_ctrl,
    kl_weight,
    rec_weight,
    factors=None,
    latent_targets=None,
    latent_align_weight=0.0,
):
    rec = poisson_loss(rates, x)
    total = rec_weight * rec + kl_weight * (kl_ic + kl_ctrl)

    if (
        factors is not None
        and latent_targets is not None
        and latent_align_weight > 0.0
    ):
        lat_loss = latent_mse_loss(factors, latent_targets)
        total = total + latent_align_weight * lat_loss

    return total, rec
