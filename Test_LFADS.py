# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 14:43:43 2025

@author: joeyl
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from Loss_Function_LFADS import lfads_loss

def test_lfads(model, test_data, batch_size=64, device=None):
    """
    Evaluates LFADS reconstruction + KL losses and returns inferred rates + factors.

    test_data : array (trials, time, neurons)
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    # Convert to tensor
    if not isinstance(test_data, torch.Tensor):
        test_data = torch.tensor(test_data, dtype=torch.float32)

    test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_rec = 0.0
    total_kl_ic = 0.0
    total_kl_ctrl = 0.0

    all_rates = []
    all_factors = []

    with torch.no_grad():
        for (xb,) in test_loader:

            xb = xb.to(device).float()          # shape (batch, T, neurons)
            xb_perm = xb.permute(0, 2, 1)       # match training: (batch, neurons, T)

            # forward pass
            rates, kl_ic, kl_ctrl, factors = model(xb_perm)

            # use kl_weight = 1.0 during evaluation
            loss, rec = lfads_loss(
                rates, xb_perm, kl_ic, kl_ctrl, kl_weight=1.0
            )

            total_loss += loss.item()
            total_rec += rec.item()
            total_kl_ic += kl_ic.item()
            total_kl_ctrl += kl_ctrl.item()

            all_rates.append(rates.cpu())
            all_factors.append(factors.cpu())

    # concatenate outputs
    rates = torch.cat(all_rates, dim=0)      # (trials, time, neurons)
    factors = torch.cat(all_factors, dim=0)  # (trials, time, factor_dim)

    N = len(test_loader)

    print("========================================")
    print(" LFADS TEST PERFORMANCE")
    print("========================================")
    print(f"Total Loss      : {total_loss / N:.4f}")
    print(f"Reconstruction  : {total_rec / N:.4f}")
    print(f"KL (IC)         : {total_kl_ic / N:.4f}")
    print(f"KL (Controller) : {total_kl_ctrl / N:.4f}")
    print("========================================")

    return rates.numpy(), factors.numpy()
