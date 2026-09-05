# Anything else

## Run it

```bash
pip install -r requirements.txt

python pipeline/scripts/run_pipeline.py --videos /path/to/folder \
    --encoder google/siglip2-base-patch16-224 --short-side 256 \
    --probe pipeline/d1_probe.npz --head pipeline/head_corrected.pt \
    --d3-path onset --name-with-probe --d2-cascade --out submission.json
```

One command, one pass, no labels, no config, no network at runtime.
SigLIP2 weights (1.5 GB) download once and can be baked into an image.

## Deploying on a live stream

| | |
|---|---|
| Ships | 2.4 MB of weights + 470 KB of code |
| Downloads once | 1.5 GB SigLIP2 |
| GPU | RTX A2000 12 GB is enough — 18–58× realtime |
| Frame source | `cv2.VideoCapture` already accepts `rtsp://` |
| Causal? | Yes — EWMA baseline and hysteresis use no future frames |
| Gap | D1's probe averages the whole clip; needs a rolling window |

Latency: D3 sub-second, D2 ~2 s (its window is centred on now).

## Known limitations

- **Boundaries are the main loss.** At tIoU 0.5 a correct class with the wrong
  extent scores zero *and* is charged twice — as a false alarm and as a miss.
- **D3 is now the weakest tier** at 42.3%: 12 predictions for 6 real events.
- **One sink class.** `road_spill_or_debris`: 0 correct, 9 false. Sharpening its
  prompts made it worse, so the class has no distinctive frame-level signature.
- **The trained head cannot localise** — 0/12 on D2 on its own. It survives in
  the pipeline only as a video-level gate, which is what it is actually good at.
- **Frame features can't express duration.** Loitering, fighting and stalled
  vehicles need motion. A video-native encoder (Cosmos-Embed1) scored *worse*.

## With more time

1. **Boundary regressor** — predict start/end offsets from frozen features
   instead of reading them off a threshold. Everything else is downstream.
2. **Region-level features** — our misses are events small in frame (congestion
   on the far carriageway, fire in one corner). CVPR 2026 work reports
   DINOv2-style features beating CLIP here; multi-crop is the cheap version.
3. **Extend the cascade to D3** — the same whether/when split that took D2 from
   14.0 to 21.3 has not been tried on long video.
4. **Per-class thresholds** — one margin of 0.15 governs all 11 classes.

## What we learned

**Local validation lied, five times.** A marks formula said 29.9 where the truth
was 51.9. Train accuracy said +27 where the truth was −24. Validation IoU rising
0.66 → 0.86 came with a *worse* score. Ground truth was revised mid-event, so
every cached label was stale.

**Looking at the frames replaced it.** Rendering the frame at the midpoint of
every predicted event into one labelled sheet found four real bugs in ten
minutes: a clip called `wrong_way_driving` six times that was queued trucks; a
crash compilation typed `stalled_vehicle` with the "CAR CRASHES TIME" watermark
in shot; two clips called normal that plainly weren't; and a correct `fire` call
destroyed by a later change. Pixels don't go stale.

**Every gain came from routing, not from a bigger model.** The two changes that
moved the score — the confidence gate and the head/onset cascade — are both
rules about *which component is allowed to speak about what*. No new weights, no
retraining, ~15 lines of code between them.

**Fitted parameters never transferred.** Every attempt to tune a threshold to the
data made things worse, and the leaderboard punished all four.

## Provenance

Every submitted JSON came from one full pipeline run — never assembled or
hand-edited. Verifiable: each prediction carries a wall-clock time measured
inside the loop, and all 28 match the run log line for line.

```
eval_casc.json: 28/28 per-video runtimes match the log | 10,066 frames encoded
```

We also dropped a rule worth 11 marks on the earlier public pack ("keep the last
interval") because it only worked by picking the right event in four known
videos. The judging runs the pipeline, not the JSON.
