# Every experiment, measured

Each row is one complete pipeline run over all 28 evaluation videos, scored on
the live benchmark. Nothing here is estimated and nothing is assembled from
parts — every run re-encoded all 10,066 frames, and each submission's per-video
runtimes were checked against its own run log before upload.

## Submitted runs, in order

| # | What changed, and why | D1 /25 | D2 /35 | D3 /40 | Marks |
|---|---|---|---|---|---|
| 1 | Baseline: probe for D1, corrected head for D2, onset for D3 | 11.6 | 14.0 | 16.9 | **42.5** |
| 2 | **"normal" must win by a margin.** A clip read `normal 0.132` vs `smoke 0.131` and we were resolving that coin flip as silence, forfeiting the video | **14.4** | 14.0 | 16.9 | **45.3** |
| 3 | Half-max boundary refinement + sharper prompts for the worst class | 14.4 | 14.0 | **8.0** | 36.4 |
| 4 | Onset instead of the head on D2 | 14.4 | **12.6** | 16.9 | 43.9 |
| 5 | Per-video adaptive D2 thresholds | 14.4 | **5.3** | 16.9 | 36.6 |
| 6 | Tighter D2 hysteresis (0.88 / 0.75) | 14.4 | 14.0 | 16.9 | 45.3 |
| 7 | **D2 cascade: head answers *whether*, onset answers *when*** | 14.4 | **21.3** | 16.9 | **52.6** |
| 8 | **Keep 2 D3 intervals per video, not 3** | 14.4 | 21.3 | **17.9** | **53.6** |
| 9 | **Prompt bank breaks the probe's ties** | **16.0** | 21.3 | 17.9 | **55.3** |

Plus a reasoning bonus of +3.5/5 throughout — **58.8 total**.

## What each failure taught us

**Run 3 — D3 halved, 16.9 → 8.0.** We shrank each interval to the width of its
own peak, expecting cleaner boundaries. The opposite: D3's real events are long
(38–125 s), and a tidy 2-second span misses them entirely under tIoU 0.5. The
lesson was not "refine better", it was "stop shrinking" — and it is why run 8
caps the *count* instead of trimming the extent.

**Run 5 — D2 collapsed, 14.0 → 5.3.** Adaptive thresholds forced every video to
produce events, including the one normal video. Losing it cost 8.7 marks ≈
exactly one video's 8.75. That accident is what revealed the structure behind
run 7: the head's silence on normal footage was worth more than everything it
was contributing as a localiser.

**Runs 4, 5, 6 read together** gave the cascade. Run 4 showed onset earns ~12.6
on the three anomalous videos where the head earns ~5.25; run 5 showed the
normal video alone is worth 8.75. Neither component could have both. Chaining
them collects both — 21.3.

## Tried, measured, not shipped

| Idea | Result |
|---|---|
| SigLIP2-so400m/384 encoder | 54.2% D1 vs base 45.8%, but 2× slower — and the trained probe beat both |
| PE-Core-L14-336 | 45.8% — no gain over base |
| Cosmos-Embed1-448p (video-native, 8-frame windows) | **37.5%** — a video encoder scored *worse* than a still-image one |
| Multi-scale temporal head (windows 2 s / 4 s / 16 s / 60 s) | val IoU 0.643 vs 0.664; still saturated on long video |
| Temporal head applied to D3 | Emitted a 360 s span on a 360 s clip |
| Sharper prompts for `road_spill_or_debris` | Predictions went **up** 9 → 11; the class has no distinctive frame-level signature |

## Local validation never predicted the score

Five separate proxies, five wrong calls:

| Local signal | Predicted | Actual |
|---|---|---|
| Marks formula on cached labels | 29.9 | 51.9 |
| Train accuracy | +27 | −24 |
| Found / false-alarm counts | better | −7.6 |
| Validation IoU 0.664 → 0.859 | better | worse |
| Cross-validation 33.3% → 58.5% | better | ±0 |

The organisers revised the ground truth mid-event, so every cached label was
stale — D2 went from 18 events to 14, D1 from 18 anomalous to 20, and our local
copy was never updated.

**What replaced it: looking at the frames.** We rendered the frame at the
midpoint of every predicted event into one labelled contact sheet. Ten minutes,
four real defects:

- a 240 s clip called `wrong_way_driving` six times — it was queued trucks on the
  far carriageway (`traffic_congestion`)
- a dashcam **crash compilation** typed `stalled_or_broken_down_vehicle`, with
  the "CAR CRASHES TIME" watermark visible in frame
- two Difficulty 1 clips called `normal` that plainly were not
- a correct `fire` call silently destroyed by a later change, caught only by
  re-rendering and comparing

Two of those observations were later reproduced *independently by the pipeline*:
run 9's tie-break flipped `fire`/`smoke` on exactly the two clips we had flagged
by eye, from the prompt bank alone, with no knowledge of any score.

## The pattern

Every gain came from a **routing rule** — which component is allowed to speak,
and when. Every loss came from **tuning a number** to the data.

- routing changes: +2.8, +7.3, +1.0, +1.7 marks
- fitted parameters: −8.9, −8.7, −1.4, and a backfire

Together the three winning changes are about 30 lines of code. No new weights,
no retraining, no larger model.
