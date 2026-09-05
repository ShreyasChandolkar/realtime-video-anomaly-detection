# AHC Visual Intelligence Hackathon — final submission

Real-time video anomaly detection, 11 classes, open-set.

**58.8 / 100** — 55.3 marks + 3.5 reasoning bonus
D1 16.0/25 (64.1%) · D2 21.3/35 (60.9%) · D3 17.9/40 (44.8%)

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
    --d3-path onset --name-with-probe --d2-cascade --d3-cap 2 \
    --out submission.json
```

Two flags carry most of the score and must not be omitted:

| flag | what it does | worth |
|---|---|---|
| `--d2-cascade` | head decides *whether*, onset decides *when* | D2 14.0 → 21.3 |
| `--d3-cap 2` | keep 2 intervals per long video, not 3 | D3 16.9 → 17.9 |

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
| `best_submission_58.8.json` | the exact file this scored on |
| `docs/slides.html` · `slides.pdf` · `AHC_slides.pptx` | two-slide presentation |
| `docs/REPO_README.md` | architecture and negative results |
| `docs/anything-else.md` | limitations, deployment, what we would do next |

## Provenance

Every submitted JSON came from **one complete pipeline run** over the real
videos — never assembled, never merged, never hand-edited. Verifiable rather
than asserted: each prediction carries a wall-clock time measured inside the
loop, and all 28 match the run log line for line.

```
eval_final2.json: 28/28 per-video runtimes match the log | 10,066 frames encoded
```

We also dropped a rule worth 11 marks on the earlier public pack ("keep the last
interval") because it worked only by picking the right event in four known
videos. The judging runs the pipeline, not the JSON.
