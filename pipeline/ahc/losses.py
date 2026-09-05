"""Asymmetric loss.

Our head reports p>0.99 across entire ten-minute videos while its training
sequences carry events on roughly 19% of frames. Plain BCE treats a confident
false positive and a confident false negative alike, and with negatives
dominating, the cheapest way for the model to avoid missing anything is to stay
permanently on.

Asymmetric loss breaks that symmetry: gamma_neg down-weights easy negatives so
they stop drowning the gradient, and the probability clip discards negatives the
model already has right, focusing what is left on the confident mistakes.

Ridnik et al., "Asymmetric Loss For Multi-Label Classification" (ICCV 2021).
"""
from __future__ import annotations


def asymmetric_loss(logits, targets, gamma_neg: float = 4.0,
                    gamma_pos: float = 1.0, clip: float = 0.05):
    import torch

    p = torch.sigmoid(logits)
    p_neg = (1 - p + clip).clamp(max=1.0)          # forgive near-certain negatives
    loss_pos = targets * torch.log(p.clamp(min=1e-8)) * (1 - p) ** gamma_pos
    loss_neg = (1 - targets) * torch.log(p_neg.clamp(min=1e-8)) * (1 - p_neg) ** gamma_neg
    return -(loss_pos + loss_neg).mean()
