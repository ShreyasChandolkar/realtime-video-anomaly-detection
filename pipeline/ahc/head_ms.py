"""Multi-scale temporal head.

The shipped head attends over a single 17-frame window, about 4.2 seconds at
4 Hz. The events it has to find span 1 second (a collision) to 125 seconds
(congestion building) - a hundredfold range. One fixed window cannot represent
both: too long to localise a crash, far too short to see a queue form.

Every current method in temporal action localisation handles this with a
feature pyramid rather than a single scale. ActionFormer encodes the sequence
into a multi-scale transformer before its classification and regression heads;
DE-Net learns representations of segments of different lengths specifically to
"mine complete abnormal events of various durations".

So this version runs parallel attention branches at several window sizes and
fuses them per frame. Kept small on purpose - the training set is a few hundred
clips, and the shipped single-scale head is 0.60M parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MSHeadConfig:
    dim: int = 768
    hidden: int = 160
    heads: int = 4
    # ~2s, 4s, 16s, 60s at 4 Hz. Odd so each window is centred on its frame.
    windows: tuple = (9, 17, 65, 241)
    dropout: float = 0.2
    n_classes: int = 11


def build_ms_model(cfg: MSHeadConfig, n_classes: int):
    import torch
    import torch.nn as nn

    class ScaleBranch(nn.Module):
        """Self-attention restricted to one temporal window."""

        def __init__(self, d, heads, window, dropout):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                              batch_first=True)
            self.window = window
            self.norm = nn.LayerNorm(d)

        def forward(self, x):
            T = x.shape[1]
            half = self.window // 2
            i = torch.arange(T, device=x.device)
            mask = (i[None, :] - i[:, None]).abs() > half
            if bool(mask.all()):            # window smaller than the sequence gap
                return self.norm(x)
            h, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
            return self.norm(x + h)

    class MultiScaleHead(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.hidden
            self.proj = nn.Sequential(
                nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, d), nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            self.branches = nn.ModuleList(
                [ScaleBranch(d, cfg.heads, w, cfg.dropout) for w in cfg.windows])
            # Learned per-scale weighting: the model decides which duration
            # matters for a given moment rather than us fixing it.
            self.gate = nn.Linear(d, len(cfg.windows))
            self.ff = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, d * 2), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(d * 2, d),
            )
            # Background slot at index 0, matching the long-sequence trainer:
            # frames outside an event need somewhere to put their probability
            # mass, or the classifier is forced to name something on ordinary
            # footage.
            self.cls = nn.Linear(d, n_classes + 1)
            self.anom = nn.Linear(d, 1)

        def forward(self, x):
            h = self.proj(x)
            outs = torch.stack([b(h) for b in self.branches], dim=-2)  # (B,T,S,D)
            w = torch.softmax(self.gate(h), dim=-1).unsqueeze(-1)      # (B,T,S,1)
            h = (outs * w).sum(dim=-2)
            h = h + self.ff(h)
            return self.cls(h), self.anom(h).squeeze(-1)

    return MultiScaleHead()


class MSHeadScorer:
    """Inference wrapper matching TemporalHeadScorer's interface."""

    def __init__(self, checkpoint: str, device: str | None = None):
        import numpy as np
        import torch

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.classes: list[str] = ck["classes"]
        self.cfg = MSHeadConfig(**ck["cfg"])
        self.background = bool(ck.get("background", False))
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_ms_model(self.cfg, len(self.classes))
        self.model.load_state_dict(ck["state"])
        self.model.to(self.device).eval()

    def score(self, emb):
        import numpy as np
        import torch

        with torch.inference_mode():
            x = torch.from_numpy(np.asarray(emb, dtype=np.float32))[None].to(self.device)
            cls, anom = self.model(x)
            p_anom = torch.sigmoid(anom)[0].float().cpu().numpy()
            p_cls = torch.softmax(cls, dim=-1)[0].float().cpu().numpy()
        return p_anom, p_cls[:, 1:]      # drop the background column for naming
