# AHC Visual Intelligence Hackathon — final submission

Real-time video anomaly detection, 11 classes, open-set.

**57.7 / 100** — 54.2 marks + 3.5 reasoning bonus
D1 16.0/25 (64.1%) · D2 20.5/35 (58.5%) · D3 17.7/40 (44.2%)

Scored after the organisers introduced false-positive penalties partway through
the event; see `docs/EXPERIMENTS.md`, which records both scoring regimes.

Frozen backbone, never fine-tuned. Trained weights total **2.4 MB**.
All 28 videos in **125 s** — 18–58× realtime on an RTX A2000 12 GB.

## Run it

```bash
pip install -r requirements.txt

python pipeline/scripts/run_pipeline.py \
    --videos /path/to/folder \
    --encoder google/siglip2-base-patch16-224 --short-side 256 \
    --probe pipeline/d1_probe.npz \
    --head  pipeline/head_corrected.pt \
    --d3-path onset --name-with-probe --d2-cascade --d3-cap 1 \
    --out submission.json
```

Two flags carry most of the score and must not be omitted:

| flag | what it does | worth |
|---|---|---|
| `--d2-cascade` | head decides *whether*, onset decides *when* | D2 14.0 → 21.3 |
| `--d3-cap 1` | one interval per long video — false positives are penalised | D3 false alarms 6 → 2 |

Reads no labels. Classes come from the fixed taxonomy in `ahc/submission.py`;
difficulty from a manifest if present, otherwise inferred from duration.
Tested on a renamed clip with no manifest: it inferred the tier and ran at 43×
realtime.

`--short-side 256` must match the head's training resolution. 384 px cost 8.7 marks.

## Contents

| Path | |
|---|---|
| `pipeline/` | the complete system — `ahc/` + `scripts/` + both checkpoints |
| `pipeline/head_corrected.pt` | temporal head, retrained after 108 mislabelled clips were fixed |
| `pipeline/d1_probe.npz` | linear probe, 12 classes (11 + normal) |
| `best_submission_54.2.json` | the exact file this scored on |
| `docs/slides.html` · `slides.pdf` · `AHC_slides.pptx` | two-slide presentation |
| `docs/REPO_README.md` | architecture and negative results |
| `docs/anything-else.md` | limitations, deployment, what we would do next |
| `docs/EXPERIMENTS.md` | every run, measured, including the scoring change |
| `pipeline/scripts/live_dashboard.py` | read-only live console over the eval pack |

## Provenance

Every submitted JSON came from **one complete pipeline run** over the real
videos — never assembled, never merged, never hand-edited. Verifiable rather
than asserted: each prediction carries a wall-clock time measured inside the
loop, and all 28 match the run log line for line.

```
eval_best.json: 28/28 per-video runtimes match the log | 10,066 frames encoded
```

We also dropped a rule worth 11 marks on the earlier public pack ("keep the last
interval") because it worked only by picking the right event in four known
videos. The judging runs the pipeline, not the JSON.
