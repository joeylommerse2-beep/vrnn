# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:55:18 2025

@author: joeyl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VRNN(nn.Module):
    def __init__(self, x_dim=28, h_dim=256, z_dim=16, n_layers=1):
        """
        x_dim: dimensionality of each input x_t
        h_dim: hidden size of RNN
        z_dim: latent variable size
        n_layers: number of RNN layers
        """
        super(VRNN, self).__init__()

        # ===== Encoder (Inference model): q(z_t | x_t, h_{t-1}) =====
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU()
        )
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU()
        )
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Linear(h_dim, z_dim)

        # ===== Prior network: p(z_t | h_{t-1}) =====
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Linear(h_dim, z_dim)

        # ===== Decoder (Generative model): p(x_t | z_t, h_{t-1}) =====
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()  # for normalized data [0,1]
        )

        # ===== Recurrent network =====
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers)

    def forward(self, x):
        """
        x: (seq_len, batch_size, x_dim)
        Returns:
            reconstruction, KL losses, etc.
        """
        seq_len, batch_size, _ = x.size()
        h = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=x.device)

        kld_loss = 0
        recon_loss = 0
        x_recon_seq = []

        for t in range(seq_len):
            phi_x_t = self.phi_x(x[t])

            # --- Prior ---
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = torch.exp(0.5 * self.prior_std(prior_t))

            # --- Encoder ---
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], dim=1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = torch.exp(0.5 * self.enc_std(enc_t))

            # --- Reparameterization ---
            eps = torch.randn_like(enc_std_t)
            z_t = enc_mean_t + eps * enc_std_t

            # --- Decoder ---
            phi_z_t = self.phi_z(z_t)
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], dim=1))
            x_recon_seq.append(dec_t)

            # --- RNN update ---
            rnn_input = torch.cat([phi_x_t, phi_z_t], dim=1).unsqueeze(0)
            _, h = self.rnn(rnn_input, h)

            # --- Losses ---
            recon_loss += F.binary_cross_entropy(dec_t, x[t], reduction='sum')
            kld_loss += -0.5 * torch.sum(1 + (2*torch.log(enc_std_t)) -
                                         (enc_mean_t - prior_mean_t).pow(2) -
                                         (enc_std_t / prior_std_t).pow(2))

        return recon_loss / batch_size, kld_loss / batch_size, torch.stack(x_recon_seq)

model = VRNN(x_dim=28, h_dim=256, z_dim=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    for batch_x in dataloader:  # batch_x: (batch, seq_len, x_dim)
        batch_x = batch_x.permute(1, 0, 2).to(device)  # (seq_len, batch, x_dim)
        optimizer.zero_grad()

        recon_loss, kld_loss, _ = model(batch_x)
        loss = recon_loss + kld_loss * 0.1  # beta-VRNN weighting

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: recon={recon_loss.item():.2f}, KL={kld_loss.item():.2f}")
