# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 21:17:16 2025

@author: joeyl
"""
import torch

def train_lfads(
    model,
    train_loader,
    val_loader,
    lfads_loss,
    epochs=100,
    lr=1e-4,
    kl_start=1e-6,
    kl_end=1.0,
    kl_anneal_epochs=75,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    kl_ic_vals = []
    kl_ctrl_vals = []
    total_kl_ic, total_kl_ctrl = 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_rec, total_kl = 0, 0, 0

        kl_weight = min(kl_end, kl_start + (kl_end - kl_start) * epoch / kl_anneal_epochs)

        for xb in train_loader:
            xb = xb.to(device).float()
            print("sample shape:", xb.shape)   # should be (batch, time, neurons)

            rates, kl_ic, kl_ctrl, factors = model(xb)
            loss, rec = lfads_loss(rates, xb, kl_ic, kl_ctrl, kl_weight)
            print("rates shape:", rates.shape, "factors shape:", factors.shape)
            print("rates mean/std:", rates.mean().item(), rates.std().item())
            print("kl_ic, kl_ctrl:", kl_ic.item(), kl_ctrl.item())

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            
            # accumulate KL values
            total_kl_ic += kl_ic.item()
            total_kl_ctrl += kl_ctrl.item()

            total_loss += loss.item()
            total_rec += rec.item()
            total_kl += (kl_ic + kl_ctrl).item()
        # average per batch for this epoch
        kl_ic_vals.append(total_kl_ic / len(train_loader))
        kl_ctrl_vals.append(total_kl_ctrl / len(train_loader))

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for xb in val_loader:
                xb = xb.to(device).float()
                rates, kl_ic, kl_ctrl, _ = model(xb)
                loss, _ = lfads_loss(rates, xb, kl_ic, kl_ctrl, kl_weight)
                val_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        print(
            f"Epoch {epoch:03d} | KLw {kl_weight:.3f} | "
            f"Train loss {train_losses[-1]:.3f} | Val loss {val_losses[-1]:.3f}"
        )

    return model, (train_losses, val_losses), (kl_ic_vals, kl_ctrl_vals)

