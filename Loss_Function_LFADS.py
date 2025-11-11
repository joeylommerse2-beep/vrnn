# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 10:06:39 2025

@author: joeyl
"""

def poisson_loss(rates, x):
    return (rates - x * torch.log(rates + 1e-8)).sum(dim=(1,2)).mean()


def lfads_loss(rates, x, kl_ic, kl_ctrl, kl_weight):
    rec = poisson_loss(rates, x)
    total = rec + kl_weight * (kl_ic + kl_ctrl)
    return total, rec
