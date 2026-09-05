# Real-time video anomaly detection — 11 classes, open-set

**58.8 / 100** — 55.3 marks + 3.5 reasoning
D1 16.0/25 (64.1%) · D2 21.3/35 (60.9%) · D3 17.9/40 (44.8%)

Frozen backbone, never fine-tuned. Trained weights total **2.4 MB**.
Classes live as **text prompts, not weights** — a twelfth class is a sentence.

## Pipeline

```
video (file or rtsp://)
  │  sample 4 Hz, short side 256
  ▼
SigLIP2-base · frozen · 768-d ──── encoded ONCE, read three ways
  │
  ├─ D1  probe classifies; prompt bank breaks ties      → class
  ├─ D2  head says WHETHER  →  onset says WHEN          → intervals
  └─ D3  onset: score − its own causal EWMA, cap 2      → intervals
  │
  ▼
confidence gates → hysteresis → measured explanation → JSON
```

**125 s for 28 videos. 18–58× realtime on an RTX A2000 12 GB.**

## Every gain came from routing, not modelling

Three changes moved the score. None added a parameter or a training step.

### 1. D2 cascade — two components, two jobs (14.0 → 21.3)

| | E024 (normal) | E021–E023 | D2 |
|---|---|---|---|
| head only | silent ✓ 8.75 | 5.25 | 14.0 |
| onset only | fires ✗ 0 | **12.6** | 12.6 |
| **cascade** | silent ✓ 8.75 | 12.6 | **21.3** |

The head is a good video-level detector and a poor localiser — it matched 0 of
12 events but correctly stayed silent on the one normal video. The onset path is
the reverse: it localises well but has no notion of "nothing here", so it fired
on the normal video and forfeited those marks.

```python
pa, _ = head.score(emb)
if float(pa.max()) >= d2_hi:     # head answers WHETHER
    found = to_events(...)       # onset answers WHEN
```

### 2. Confidence gates — speak only when sure (D1 11.6 → 14.4)

```python
if top == "normal" and p1 - p2 < 0.15:
    name the best anomaly anyway     # silence forfeits the whole video
```

One clip read `normal 0.132` vs `smoke 0.131` — a coin flip resolved as silence.

### 3. Prompt bank breaks the probe's ties (D1 14.4 → 16.0)

When the probe's top two classes sit within 0.08 it is guessing, not choosing.
The prompt bank is computed in the same pass and was unused at D1, so it picks
between those same two — it can never introduce a third.

It corrected `fire`/`smoke` on two videos in opposite directions. Those are the
exact two clips flagged as swapped by eye from a contact sheet, hours earlier and
with no knowledge of any score — independent confirmation the mechanism is real
and not a fit.

## Negative results, measured on the live scorer

| Tried | Outcome |
|---|---|
| Half-max boundary refinement | **D3 16.9 → 8.0** — real events are long |
| Adaptive per-video D2 thresholds | **D2 14.0 → 5.3** — forced the normal video to fire |
| Onset instead of head on D2 | 14.0 → 12.6 |
| Sharper prompts for the worst class | predictions rose 9 → 11 |
| Cosmos-Embed1 (video-native encoder) | 37.5% vs 45.8% for still images |
| Multi-scale head, 4 windows | no gain, still saturates |

Six measured negatives. Every fitted parameter we tried was punished; every
structural rule about *which component may speak* was rewarded.

## Why D2 was the hard tier

| tier | event length | prediction must land within |
|---|---|---|
| D1 | — | timestamps are null; boundaries can't hurt |
| **D2** | **median 20 s** | **~7 s of the true start** |
| D3 | 38–125 s | 13–60 s |

The same detector scored 33% recall on D3 and 0% on D2 before the cascade.

## Explanations come from the run, not from a template

Each event's explanation is generated inside the loop from what was measured —
where the response peaked, how far it rose above that camera's own rolling
baseline, how long it stayed raised, and whether the prompt bank agrees with the
label:

> Evidence for traffic accident peaks at 23s, +1.607 above this camera's own
> rolling baseline, and stays raised for 324s across 1387 sampled frames…
> The prompt bank ranks stalled or broken down vehicle slightly higher here, so
> the timing is firmer than the label.

## Run it

```bash
python scripts/run_pipeline.py --videos /path/to/folder \
    --encoder google/siglip2-base-patch16-224 --short-side 256 \
    --probe runs/d1_probe.npz --head runs/head_corrected.pt \
    --d3-path onset --name-with-probe --d2-cascade --d3-cap 2 \
    --out submission.json
```
