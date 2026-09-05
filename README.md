# AHC Visual Intelligence Hackathon — final submission

Real-time video anomaly detection, 11 classes, open-set.

**52.6 / 100** · D1 14.4/25 · D2 21.3/35 · D3 16.9/40 · reasoning +3.5/5

Frozen backbone, never fine-tuned. Trained weights total **2.4 MB**.
Full 28-video pack in **125 s** — 18–58× realtime on an RTX A2000 12 GB.

## Run it

```bash
pip install -r requirements.txt

python pipeline/scripts/run_pipeline.py \
    --videos /path/to/folder \
    --encoder google/siglip2-base-patch16-224 --short-side 256 \
    --probe pipeline/d1_probe.npz \
    --head  pipeline/head_corrected.pt \
    --d3-path onset --name-with-probe --d2-cascade \
    --out submission.json
```

`--d2-cascade` is what took Difficulty 2 from 14.0 to 21.3. Do not omit it.

Reads no labels. Classes come from the fixed taxonomy in `ahc/submission.py`;
difficulty from a manifest if present, otherwise from duration. Tested on a
renamed single clip with no manifest: it inferred the tier and ran at 43×
realtime.

`--short-side 256` must match the head's training resolution. 384 px cost 8.7 marks.

## Contents

| Path | |
|---|---|
| `pipeline/` | the complete system — `ahc/` + `scripts/` + both checkpoints |
| `pipeline/head_corrected.pt` | temporal head, retrained after 108 mislabelled clips were fixed |
| `pipeline/d1_probe.npz` | linear probe, 12 classes (11 + normal) |
| `best_submission_52.6.json` | the exact file that scored 52.6 |
| `docs/slides.html` | two-slide presentation |
| `docs/REPO_README.md` | architecture and negative results |
| `docs/anything-else.md` | limitations, deployment, what we would do next |

## Provenance

Every submitted JSON came from **one complete pipeline run** over the real
videos — never assembled, never merged, never hand-edited. Verifiable rather
than asserted: each prediction carries a wall-clock time measured inside the
loop, and all 28 match the run log line for line.

```
eval_casc.json: 28/28 per-video runtimes match the log | 10,066 frames encoded
```

We also dropped a rule worth 11 marks on the earlier public pack ("keep the last
interval") because it worked only by picking the right event in four known
videos. The judging runs the pipeline, not the JSON.
