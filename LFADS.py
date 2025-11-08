# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 16:38:58 2025

@author: joeyl

LFADS (Latent Factor Analysis via Dynamical Systems)
---------------------------------------------------
PyTorch implementation of LFADS for neural population data.

Features:
- Encoder + Generator + Controller structure
- Poisson reconstruction loss
- KL annealing
- Visualization of latent trajectories and reconstructions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ===============================================================
# LFADS MODEL
# ===============================================================
class LFADS(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=16,
        factor_dim=8,
        controller_dim=8,
        generator_hidden=64,
        controller_hidden=64,
        encoder_hidden=64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.factor_dim = factor_dim

        # Encoder for initial condition
        self.encoder = nn.GRU(input_dim, encoder_hidden, batch_first=True, bidirectional=True)
        self.encoder_mu = nn.Linear(2 * encoder_hidden, latent_dim)
        self.encoder_logvar = nn.Linear(2 * encoder_hidden, latent_dim)

        # Controller RNN (time-varying inputs)
        self.controller = nn.GRU(input_dim + factor_dim, controller_hidden, batch_first=True)
        self.controller_mu = nn.Linear(controller_hidden, latent_dim)
        self.controller_logvar = nn.Linear(controller_hidden, latent_dim)

        # Generator RNN
        self.generator = nn.GRU(latent_dim, generator_hidden, batch_first=True)
        self.generator_to_factors = nn.Linear(generator_hidden, factor_dim)

        # Decoder (factors → rates)
        self.factors_to_rates = nn.Linear(factor_dim, input_dim)

    def encode_initial_condition(self, x):
        _, h = self.encoder(x)
        h = torch.cat([h[0], h[1]], dim=-1)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        x: (batch, time, input_dim)
        Returns:
            rates: reconstructed firing rates
            kl_ics: KL for initial conditions
            kl_ctrl: KL for controller inputs
            factors: latent low-dim trajectories
        """
        batch_size, T, _ = x.size()

        # Encode initial condition
        mu0, logvar0 = self.encode_initial_condition(x)
        z0 = self.sample(mu0, logvar0)
        g_h = torch.zeros(1, batch_size, self.generator.hidden_size, device=x.device)

        kl_ics = 0.5 * torch.sum(
            torch.exp(logvar0) + mu0**2 - 1. - logvar0, dim=1
        ).mean()

        # Initialize controller state
        c_h = torch.zeros(1, batch_size, self.controller.hidden_size, device=x.device)
        prev_factors = torch.zeros(batch_size, self.factor_dim, device=x.device)

        rates = []
        factors_all = []
        kl_ctrl_total = 0.0

        for t in range(T):
            controller_input = torch.cat([x[:, t, :], prev_factors], dim=-1).unsqueeze(1)
            _, c_h = self.controller(controller_input, c_h)

            mu_u = self.controller_mu(c_h.squeeze(0))
            logvar_u = self.controller_logvar(c_h.squeeze(0))
            u_t = self.sample(mu_u, logvar_u).unsqueeze(1)

            kl_t = 0.5 * torch.sum(
                torch.exp(logvar_u) + mu_u**2 - 1. - logvar_u, dim=1
            ).mean()
            kl_ctrl_total += kl_t

            # Generator step
            _, g_h = self.generator(u_t, g_h)
            factors = self.generator_to_factors(g_h.squeeze(0))
            prev_factors = factors

            rates_t = torch.exp(self.factors_to_rates(factors))
            rates.append(rates_t.unsqueeze(1))
            factors_all.append(factors.unsqueeze(1))

        rates = torch.cat(rates, dim=1)
        factors_all = torch.cat(factors_all, dim=1)
        kl_ctrl_total /= T

        return rates, kl_ics, kl_ctrl_total, factors_all


