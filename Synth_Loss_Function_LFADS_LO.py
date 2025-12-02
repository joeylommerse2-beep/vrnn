# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 10:06:39 2025

@author: joeyl
"""
import torch 

def poisson_loss(rates, x):
    return (rates - x * torch.log(rates + 1e-8)).sum(dim=(1,2)).mean()

def latent_mse_loss(factors, latents_true, eps=1e-8):
    """
    factors:      (B, T, F)
    latents_true: (B, T, K)  # here F == K == 3

    Z-score latents per dimension using stats from latents_true,
    then compute MSE between normalized true and normalized factors.
    This makes each dimension contribute roughly equally.
    """
    B, T, K = latents_true.shape

    # Flatten for stats
    L_flat = latents_true.reshape(B * T, K)
    mean = L_flat.mean(dim=0, keepdim=True)                 # (1, K)
    std  = L_flat.std(dim=0, keepdim=True) + eps            # (1, K)

    # Normalize both true and predicted w.r.t. true stats
    L_norm = (latents_true - mean) / std
    F_norm = (factors      - mean) / std

    return ((F_norm - L_norm) ** 2).mean()


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
