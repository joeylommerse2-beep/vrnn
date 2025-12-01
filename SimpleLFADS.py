# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 11:15:04 2025

@author: joeyl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset



class SimpleLFADS(nn.Module):
    def __init__(self, input_dim, latent_dim=3, factor_dim=3, gen_hidden=3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.factor_dim = factor_dim

        # Initial condition encoder
        self.encoder = nn.GRU(input_dim, 32, batch_first=True, bidirectional=True)
        self.encoder_mu = nn.Linear(64, latent_dim)
        self.encoder_logvar = nn.Linear(64, latent_dim)
        self.ic_to_g0 = nn.Linear(latent_dim, gen_hidden)

        # Generator: autonomous RNN
        self.generator = nn.GRU(0, gen_hidden, batch_first=True)  # input_size=0 is not allowed; see below
        self.generator_to_factors = nn.Linear(gen_hidden, factor_dim)

        self.factors_to_rates = nn.Linear(factor_dim, input_dim)

    def encode_ic(self, x):
        _, h = self.encoder(x)
        h = torch.cat([h[0], h[1]], dim=-1)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def forward(self, x):
        B, T, _ = x.size()

        mu0, logvar0 = self.encode_ic(x)
        z0 = mu0  # no sampling for this synthetic test
        g0 = torch.tanh(self.ic_to_g0(z0)).unsqueeze(0)  # (1, B, gen_hidden)

        # Autonomous generator: feed zeros as dummy input
        u = torch.zeros(B, T, 1, device=x.device)
        g_seq, _ = self.generator(u, g0)

        factors = self.generator_to_factors(g_seq)  # (B, T, factor_dim)
        rates = torch.exp(self.factors_to_rates(factors)) + 1e-6

        # Ignore KL completely
        kl_ic = torch.tensor(0., device=x.device)
        kl_ctrl = torch.tensor(0., device=x.device)

        return rates, kl_ic, kl_ctrl, factors
