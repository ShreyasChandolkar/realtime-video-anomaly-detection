# Real-time video anomaly detection — 11 classes, open-set

**52.6 / 100** · D1 14.4/25 · D2 21.3/35 · D3 16.9/40 · reasoning +3.5/5

Frozen backbone, never fine-tuned. Trained weights total **2.4 MB**.
Classes live as **text prompts, not weights** — a twelfth class is a sentence.

## Pipeline

```
video (file or rtsp://)
  │  sample 4 Hz, short side 256
  ▼
SigLIP2-base · frozen · 768-d ──── encoded ONCE, read three ways
  │
  ├─ D1  linear probe on the mean-pooled clip     → class
  ├─ D2  head says WHETHER  →  onset says WHEN    → intervals
  └─ D3  onset: prompt score − its own EWMA       → rise over baseline
  │
  ▼
confidence gates → hysteresis → intervals → explanation → JSON
```

**125 s for 28 videos. 18–58× realtime on an RTX A2000 12 GB.**

## The idea that carried the score: specialist roles

D2 was stuck at 14.0/35 with **0 of 12 events matched**. The two components each
did half the job:

| | E024 (normal) | E021–E023 | D2 |
|---|---|---|---|
| head only | silent ✓ 8.75 | 5.25 | 14.0 |
| onset only | fires ✗ 0 | **12.6** | 12.6 |
| **cascade** | silent ✓ 8.75 | 12.6 | **21.3** |

The head is a good video-level detector and a poor localiser. The onset path is
the reverse — it localises well but has no notion of "nothing here", so it fired
on the normal video and forfeited a whole video's marks. Cascading them:

```python
pa, _ = head.score(emb)
if float(pa.max()) >= d2_hi:     # head answers WHETHER
    found = to_events(...)       # onset answers WHEN
```

E021 changed from `road_spill_or_debris` to `traffic_congestion` — the class
confirmed by eye from the frames. E023 became three real spans instead of one
240 s blanket. E024 stayed silent.

## The other idea: speak only when confident

```python
if top == "normal" and p1 - p2 < 0.15:
    name the best anomaly anyway     # silence forfeits the whole video
```

One clip read `normal 0.132` vs `smoke 0.131`. Gating that coin flip took
**D1 from 11.6 → 14.4**. The same gate on naming fixed a crash compilation
mislabelled `stalled_vehicle` while preserving a correct `fire` call that an
ungated version destroyed.

Both ideas match the 2026 two-pass paper on this exact problem (rare traffic
events, real CCTV): separate grounding from typing, and revert on hedges. We got
there from looking at frames, then found it agreed.

## Negative results, measured on the live scorer

| Tried | Outcome |
|---|---|
| Half-max boundary refinement | **D3 16.9 → 8.0** — real events are long |
| Onset instead of head on D2 | 14.0 → 12.6 (lost the normal video) |
| Adaptive per-video D2 thresholds | 14.0 → **5.3** (forced the normal video to fire) |
| Sharper prompts for the worst class | predictions rose 9 → 11 |
| Cosmos-Embed1 (video-native encoder) | 37.5% vs 45.8% for still images |
| Multi-scale head, 4 windows | no gain, still saturates |

## Why D2 was the hard tier

| tier | event length | prediction must land within |
|---|---|---|
| D1 | — | timestamps are null; boundaries can't hurt |
| **D2** | **median 20 s** | **~7 s of the true start** |
| D3 | 38–125 s | 13–60 s |

The same detector scored 33% recall on D3 and 0% on D2. Long events forgive a
loose span; 20-second events do not.

## Run it

```bash
python scripts/run_pipeline.py --videos /path/to/folder \
    --encoder google/siglip2-base-patch16-224 --short-side 256 \
    --probe runs/d1_probe.npz --head runs/head_corrected.pt \
    --d3-path onset --name-with-probe --d2-cascade --out submission.json
```

Reads no labels. `--short-side 256` must match the head's training resolution.
