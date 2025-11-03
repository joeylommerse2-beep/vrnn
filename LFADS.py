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


# ===============================================================
# LOSS FUNCTIONS
# ===============================================================
def poisson_loss(rates, x):
    return (rates - x * torch.log(rates + 1e-8)).mean()


def lfads_loss(rates, x, kl_ic, kl_ctrl, kl_weight):
    rec = poisson_loss(rates, x)
    total = rec + kl_weight * (kl_ic + kl_ctrl)
    return total, rec


# ===============================================================
# TRAINING LOOP
# ===============================================================
def train_lfads(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-3,
    kl_start=1e-4,
    kl_end=1.0,
    kl_anneal_epochs=50,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_rec, total_kl = 0, 0, 0

        kl_weight = min(kl_end, kl_start + (kl_end - kl_start) * epoch / kl_anneal_epochs)

        for xb, in train_loader:
            xb = xb.to(device)
            rates, kl_ic, kl_ctrl, _ = model(xb)
            loss, rec = lfads_loss(rates, xb, kl_ic, kl_ctrl, kl_weight)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 200.0)
            opt.step()

            total_loss += loss.item()
            total_rec += rec.item()
            total_kl += (kl_ic + kl_ctrl).item()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for xb, in val_loader:
                xb = xb.to(device)
                rates, kl_ic, kl_ctrl, _ = model(xb)
                loss, _ = lfads_loss(rates, xb, kl_ic, kl_ctrl, kl_weight)
                val_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        print(
            f"Epoch {epoch:03d} | KLw {kl_weight:.3f} | "
            f"Train loss {train_losses[-1]:.3f} | Val loss {val_losses[-1]:.3f}"
        )

    return model, (train_losses, val_losses)


# ===============================================================
# VISUALIZATION UTILITIES
# ===============================================================
def visualize_latents(factors, title="LFADS latent trajectories"):
    """
    factors: (batch, time, factor_dim)
    Plots the first few latent factors
    """
    with torch.no_grad():
        f = factors[0].cpu().numpy()  # first trial
    plt.figure(figsize=(8, 4))
    plt.plot(f[:, :min(5, f.shape[1])])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Latent factors")
    plt.legend([f"f{i}" for i in range(min(5, f.shape[1]))])
    plt.tight_layout()
    plt.show()


def visualize_reconstruction(x, rates, neuron_idx=0):
    """
    Plots observed vs reconstructed rates for a given neuron
    """
    x = x[0].cpu().numpy()
    r = rates[0].detach().cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.plot(x[:, neuron_idx], label="Observed (spikes)")
    plt.plot(r[:, neuron_idx], label="Reconstructed rate")
    plt.title(f"Neuron {neuron_idx} Reconstruction")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===============================================================
# EXAMPLE USAGE
# ===============================================================
if __name__ == "__main__":
    # Simulated data: 200 trials, 50 neurons, 100 time bins
    N, T, D = 200, 100, 50
    data = torch.poisson(torch.rand(N, T, D) * 5.0)

    train_ds = TensorDataset(data[:160])
    val_ds = TensorDataset(data[160:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = LFADS(input_dim=D)
    trained_model, losses = train_lfads(model, train_loader, val_loader, epochs=50)

    # Visualize latent dynamics & reconstruction
    xb = data[160:161]
    with torch.no_grad():
        rates, _, _, factors = trained_model(xb)

    visualize_latents(factors)
    visualize_reconstruction(xb, rates)
