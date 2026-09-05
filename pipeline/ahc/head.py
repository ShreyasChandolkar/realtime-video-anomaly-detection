"""A small learned temporal head over frozen SigLIP2 features.

Every current method in this space puts a lightweight learned module on top of
frozen CLIP frame features - VadCLIP uses windowed self-attention plus a graph
layer, CLIP-TSA uses temporal self-attention, others use a small LSTM. We were
the exception: hand-designed EWMA + hysteresis where the field learns it.

Two reasons this is expected to transfer where our threshold tuning did not:

  - The output is a calibrated probability, so 0.5 means the same thing on any
    video. Our raw cosine margins shifted scale whenever the prompt bank
    changed, which silently invalidated every fitted threshold.
  - Attention is computed inside a sliding window, so a clip of 24 frames and a
    video of 2400 see the same local context. Nothing depends on video length.

Deliberately small (~0.5M parameters). The training set is 484 clips; anything
larger memorises it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HeadConfig:
    dim: int = 768           # SigLIP2-base embedding width
    hidden: int = 192
    heads: int = 4
    window: int = 17         # frames of context, ~4s at 4Hz (odd, centred)
    dropout: float = 0.2
    n_classes: int = 11      # anomaly classes; "normal" is the absence of all


def build_model(cfg: HeadConfig, n_classes: int):
    """Windowed temporal attention -> per-frame class logits + anomaly logit."""
    import torch
    import torch.nn as nn

    class WindowedAttention(nn.Module):
        """Self-attention restricted to a local window.

        A global attention would let a 10-minute video attend across minutes,
        which both costs O(T^2) and makes the model's behaviour depend on
        length. Restricting it keeps cost linear and behaviour identical for
        short clips and long footage.
        """

        def __init__(self, d, heads, window, dropout):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                              batch_first=True)
            self.window = window
            self.norm = nn.LayerNorm(d)

        def forward(self, x):                       # x: (B, T, D)
            T = x.shape[1]
            half = self.window // 2
            i = torch.arange(T, device=x.device)
            mask = (i[None, :] - i[:, None]).abs() > half   # True = blocked
            h, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
            return self.norm(x + h)

    class TemporalHead(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.hidden
            self.proj = nn.Sequential(
                nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, d), nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            self.attn1 = WindowedAttention(d, cfg.heads, cfg.window, cfg.dropout)
            self.attn2 = WindowedAttention(d, cfg.heads, cfg.window, cfg.dropout)
            self.ff = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, d * 2), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(d * 2, d),
            )
            # Class head carries an explicit background slot at index 0.
            # Without it the classifier must name something on every frame, so
            # ordinary footage gets forced into whichever class fits least
            # badly - which is how a quiet motorway becomes "wrong way driving".
            self.cls = nn.Linear(d * 2, n_classes + 1)
            self.anom = nn.Linear(d * 2, 1)

        def forward(self, x):                       # (B, T, 768)
            h = self.proj(x)
            h = self.attn1(h)
            h = self.attn2(h)
            h = h + self.ff(h)
            # Max and mean over the whole sequence, broadcast back per frame.
            # Mean alone dilutes transient events - an accident lasts about a
            # second in a ten-minute video - so the peak is carried alongside
            # the context rather than averaged into it.
            ctx = h.max(dim=1, keepdim=True).values + h.mean(dim=1, keepdim=True)
            hh = torch.cat([h, ctx.expand_as(h)], dim=-1)
            return self.cls(hh), self.anom(hh).squeeze(-1)

    return TemporalHead()


class TemporalHeadScorer:
    """Inference wrapper: features in, per-frame anomaly probability and class."""

    def __init__(self, checkpoint: str, device: str | None = None):
        import torch

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.classes: list[str] = ck["classes"]
        self.background: bool = bool(ck.get("background", False))
        self.cfg = HeadConfig(**ck["cfg"])
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.cfg, len(self.classes))
        self.model.load_state_dict(ck["state"])
        self.model.to(self.device).eval()

    def score(self, emb: np.ndarray, return_logits: bool = False):
        """Per-frame anomaly probability and class probabilities.

        With return_logits, also returns the raw anomaly logit. That matters on
        long video: once sigmoid saturates at 0.999 it has destroyed the
        variation underneath, so a baseline-relative signal computed on the
        probability is differencing a flat line. The logit keeps moving.
        """
        import torch

        with torch.inference_mode():
            x = torch.from_numpy(np.asarray(emb, dtype=np.float32))[None].to(self.device)
            cls, anom = self.model(x)
            logit = anom[0].float().cpu().numpy()
            p_anom = torch.sigmoid(anom)[0].float().cpu().numpy()
            p_cls = torch.softmax(cls, dim=-1)[0].float().cpu().numpy()
            if self.background:          # drop the background column for naming
                p_cls = p_cls[:, 1:]
        if return_logits:
            return p_anom, p_cls, logit
        return p_anom, p_cls
