# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 08:51:53 2025

@author: joeyl
"""

import torch
import math

def generate_latents(T=80, K=3, n_trials=200, device="cpu"):
    """
    Returns latents of shape (n_trials, T, K)
    """
    t = torch.linspace(0, 1.0, T, device=device)  # normalized time [0,1]

    # Define 3 base latent patterns (T,)
    f1 = torch.sin(2 * math.pi * 1 * t)          # 1 Hz-like
    f2 = torch.sin(2 * math.pi * 2 * t + 0.5)    # 2 Hz + phase
    f3 = torch.cos(2 * math.pi * 1.5 * t)        # 1.5 Hz cos

    F_base = torch.stack([f1, f2, f3], dim=-1)   # (T, 3)

    # Trial-specific scaling + noise
    latents = []
    for _ in range(n_trials):
        # random amplitude per factor
        amps = torch.randn(K, device=device) * 0.5 + 1.0  # around 1, some variation
        L = F_base * amps  # (T, K)

        # small additive Gaussian noise
        L = L + 0.1 * torch.randn_like(L)

        latents.append(L)

    latents = torch.stack(latents, dim=0)  # (n_trials, T, K)
    return latents

def latents_to_rates(latents, n_neurons=42, device="cpu"):
    """
    latents: (n_trials, T, K)
    returns rates: (n_trials, T, n_neurons)
    """
    n_trials, T, K = latents.shape

    # Random mixing matrix W: (K, n_neurons)
    W = torch.randn(K, n_neurons, device=device) * 0.5
    # Bias per neuron, so average firing is positive and reasonable
    b = torch.randn(n_neurons, device=device) * 0.2 + 1.0

    # Flatten latents over trials and time
    L_flat = latents.reshape(-1, K)                     # (n_trials*T, K)
    H_flat = L_flat @ W + b                             # (n_trials*T, n_neurons)

    # Nonlinearity: softplus or exp, but not too extreme
    rates_flat = torch.exp(H_flat)   # (n_trials*T, n_neurons)

    rates = rates_flat.reshape(n_trials, T, n_neurons)  # (n_trials, T, n_neurons)
    return rates, W, b

def rates_to_spikes(rates, dt=0.01):
    """
    rates: (n_trials, T, n_neurons), in Hz
    returns spike counts: (n_trials, T, n_neurons) as integers
    """
    lam = rates * dt
    # Poisson sampling
    spikes = torch.poisson(lam)
    return spikes

def make_synthetic_dataset(
    n_trials=200,
    T=80,
    K_true=3,
    n_neurons=42,
    device="cpu"
):
    latents = generate_latents(T=T, K=K_true, n_trials=n_trials, device=device)
    rates, W, b = latents_to_rates(latents, n_neurons=n_neurons, device=device)
    spikes = rates_to_spikes(rates, dt=0.01)
    return spikes, latents, rates, W, b
