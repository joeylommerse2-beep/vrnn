# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 10:06:39 2025

@author: joeyl
"""
import torch 

def poisson_loss(rates, x):
    return (rates - x * torch.log(rates + 1e-8)).sum(dim=(1,2)).mean()

def latent_alignment_loss(factors, latents_true, eps=1e-6):
    """
    factors: (B, T, F)
    latents_true: (B, T, K)

    We compute a least-squares linear map A (F x K) such that:
        F_flat @ A ≈ L_flat
    and penalize the MSE between F_flat @ A and L_flat.

    This encourages the factors to span a space from which the true latents
    can be linearly reconstructed.
    """
    B, T, F = factors.shape
    _, _, K = latents_true.shape

    F_flat = factors.reshape(B * T, F)        # (N, F)
    L_flat = latents_true.reshape(B * T, K)   # (N, K)

    # Center to remove trivial offsets
    F_flat = F_flat - F_flat.mean(dim=0, keepdim=True)
    L_flat = L_flat - L_flat.mean(dim=0, keepdim=True)

    # Solve (F^T F) A = F^T L  ⇒  A = (F^T F)^-1 F^T L
    FtF = F_flat.T @ F_flat + eps * torch.eye(F, device=factors.device)
    FtL = F_flat.T @ L_flat

    A = torch.linalg.solve(FtF, FtL)          # (F, K)
    L_pred = F_flat @ A                       # (N, K)

    return torch.mean((L_pred - L_flat) ** 2)


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
    """
    Extended LFADS loss for synthetic debugging:

    - Always includes Poisson reconstruction + KL (if kl_weight > 0).
    - Optionally includes a supervised latent-alignment term if
      latent_targets is provided and latent_align_weight > 0.
    """
    rec = poisson_loss(rates, x)
    total = rec_weight * rec + kl_weight * (kl_ic + kl_ctrl)

    if (
        factors is not None
        and latent_targets is not None
        and latent_align_weight > 0.0
    ):
        lat_loss = latent_alignment_loss(factors, latent_targets)
        total = total + latent_align_weight * lat_loss

    return total, rec
