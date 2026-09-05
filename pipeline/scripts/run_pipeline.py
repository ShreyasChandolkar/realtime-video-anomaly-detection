#!/usr/bin/env python3
"""Run the detector on a folder of video and write a submission. No labels.

This is the entry point for evaluation on unseen data, and it is deliberately
separate from the development scripts, which all read ground_truth.csv for the
class list and the difficulty of each video. Neither exists at evaluation time,
so both are derived here instead:

  classes     the fixed 11-class taxonomy, not whatever a CSV happens to contain
  difficulty  from the manifest when given, otherwise from video duration - a
              short clip carries one event and wants a video-level answer, a
              long recording wants intervals

Nothing here is fitted to a particular video. Two selection rules were tested on
the public pack and only one survives that standard:

  keep the N longest intervals   KEPT. Under a tIoU-0.5 rule a spurious short
                                 interval is pure loss - it cannot match
                                 anything and it spends precision. Emitting
                                 every interval scored 12.1 on Difficulty 3
                                 against 19.5 when capped. Duration is a
                                 property of the detection, not of the video.

  keep the *last* interval       DROPPED. It scored well because it happened to
                                 pick the right event in four known videos.
                                 On unseen footage "last" means nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ahc import features, prompts
from ahc.branding import PUBLIC_MODEL_NAME
from ahc.postproc import FAMILIES, HysteresisTracker, to_events
from ahc.stream import StreamConfig, StreamScorer
from ahc.submission import CLASSES, PredictedEvent, Submission, VideoRuntime

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

# Shipped settings. Not tuned per dataset.
D1_THRESHOLD = 0.20
D2_HI, D2_LO = 0.80, 0.40
D3_HI, D3_LO = 0.10, 0.04
D3_CAP = 3                      # longest intervals per video; see the docstring
# How much better than its own runner-up the probe must be before it is allowed
# to rename an interval the detector already found. Below this it is guessing
# between two classes, and the detector's answer is the better bet.
PROBE_MIN_MARGIN = 0.15
# The same bar, applied to silence. Calling a Difficulty 1 clip normal forfeits
# the whole video if it is not, so "normal" has to win by a real margin rather
# than by a hair - it led 0.132 to 0.131 on a clip showing a car spun across the
# track, and that coin flip was being resolved as "say nothing".
NORMAL_MIN_MARGIN = 0.15
# An interval is kept only if its peak reaches this fraction of the strongest
# peak in the same video. The cap below is a ceiling, not a quota: emitting a
# fixed three intervals per video means the weakest is filler, and under
# tIoU 0.5 filler is charged twice - once as a false alarm, once for the real
# event it fails to cover.
PEAK_KEEP_FRACTION = 0.5
# Window used to smooth a score before its width is measured. Long enough to
# bridge the frame-to-frame noise in the onset signal, short enough that a brief
# event still has an envelope of its own.
SMOOTH_SECONDS = 8.0
# Difficulty from duration when the manifest does not carry it. The tiers below
# match how the task is posed rather than any particular dataset: a short clip
# holds one event and wants a video-level answer; a few minutes wants intervals;
# long footage wants the onset path, which is built for events occupying a small
# fraction of a long timeline.
SHORT_VIDEO_SECONDS = 60.0      # below this -> Difficulty 1
LONG_VIDEO_SECONDS = 300.0      # above this -> Difficulty 3


def refine(times, score, s: float, e: float) -> tuple[float, float]:
    """Shrink a detected span to the width of its own peak, at half maximum.

    Hysteresis returns the span where the score sat above a low exit threshold,
    which on long footage runs far past the event - one span covered 7s to 331s
    of a 602s video, and another covered a whole 240s clip. tIoU 0.5 refuses
    both however well centred they are: a 41s event inside a 240s claim scores
    0.17. Half-maximum width is the standard way to read a peak's extent, and it
    keeps the moment the detector actually responded to rather than the long
    tail either side of it.

    The full-width-half-maximum idea is the same one the 2026 two-pass grounding
    work applies as a second refinement pass: locate coarsely, then tighten.
    """
    m = (times >= s) & (times <= e)
    if not m.any():
        return s, e
    idx = np.where(m)[0]
    seg = score[idx]
    # Measure the width of the envelope, not of the spike. The semantic onset
    # score is noisy frame to frame, and half-maximum on the raw signal returns
    # the peak itself - it cut a 67s span to 2s and an 18s fire to 1s, which
    # tIoU 0.5 rejects just as firmly as the 331s span it was meant to fix.
    w = max(3, int(round(SMOOTH_SECONDS * len(seg) /
                         max(float(times[idx[-1]] - times[idx[0]]), 1e-6))))
    if w % 2 == 0:
        w += 1
    if len(seg) > w:
        pad = np.pad(seg, w // 2, mode="edge")
        seg = np.convolve(pad, np.ones(w) / w, mode="valid")[:len(idx)]
    k = int(np.argmax(seg))
    floor = float(seg.min())
    half = floor + 0.5 * (float(seg[k]) - floor)
    i = k
    while i > 0 and seg[i - 1] >= half:
        i -= 1
    j = k
    while j < len(seg) - 1 and seg[j + 1] >= half:
        j += 1
    a, b = float(times[idx[i]]), float(times[idx[j]])
    return (a, b) if b > a else (s, e)


def discover(root: Path) -> list[Path]:
    d = root / "videos" if (root / "videos").is_dir() else root
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)


def levels_from_manifest(root: Path) -> dict[str, int]:
    """Use a provided difficulty if there is one; never invent labels."""
    import pandas as pd
    for name in ("videos.csv", "manifest.csv"):
        p = root / name
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        idc = cols.get("video_id") or cols.get("id")
        lvc = cols.get("level") or cols.get("difficulty")
        if idc and lvc:
            return {str(r[idc]): int(r[lvc]) for _, r in df.iterrows()
                    if str(r[lvc]).strip().isdigit()}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder of video, or a parent with videos/")
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--cache", default="", help="feature cache; a temp dir if unset")
    ap.add_argument("--encoder", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--base-encoder", default="google/siglip2-base-patch16-224")
    ap.add_argument("--head", default="", help="temporal head checkpoint for D2")
    ap.add_argument("--probe", default="", help="trained D1 probe (npz); falls back to prompts")
    ap.add_argument("--head-ms", default="", help="multi-scale head for D3 long video")
    ap.add_argument("--name-with-probe", action="store_true",
                    help="name D2/D3 intervals with the trained probe instead of "
                         "prompt matching or the head's own classifier")
    ap.add_argument("--refine", action="store_true",
                    help="shrink spans to half-max width. OFF by default: it cut "
                         "Difficulty 3 from 16.9 to 8.0, because the events there "
                         "are long and a tidy span misses them under tIoU 0.5")
    ap.add_argument("--d2-hi", type=float, default=D2_HI)
    ap.add_argument("--d2-lo", type=float, default=D2_LO,
                    help="hysteresis exit for D2. Higher = tighter spans. D2 "
                         "events run ~20s, so a prediction must land within ~7s "
                         "of the true start to reach tIoU 0.5; the default exit "
                         "produces 18-44s spans, roughly twice too wide")
    ap.add_argument("--d2-cascade", action="store_true",
                    help="head decides IF the video is anomalous, onset decides "
                         "WHEN. Each component is used for the job it is good at")
    ap.add_argument("--d2-adaptive", action="store_true",
                    help="take D2 thresholds from quantiles of the video's own "
                         "p(anomaly) instead of fixed 0.80/0.40")
    ap.add_argument("--d2-cap", type=int, default=D3_CAP,
                    help="max intervals per D2 video in cascade mode. Separate "
                         "from --d3-cap: the two tiers have different event "
                         "densities and must be capped independently")
    ap.add_argument("--d3-cap", type=int, default=D3_CAP,
                    help="max intervals kept per long video")
    ap.add_argument("--merge-gap", type=float, default=0.0,
                    help="join same-class intervals separated by less than this "
                         "many seconds")
    ap.add_argument("--d3-path", default="onset", choices=("onset", "head", "ms"),
                    help="which detector Difficulty 3 uses")
    ap.add_argument("--sample-hz", type=float, default=4.0)
    ap.add_argument("--short-side", type=int, default=256,
                    help="decode resolution; must match what the head was trained on")
    ap.add_argument("--device", default=None)
    ap.add_argument("--explain", action="store_true", default=True)
    a = ap.parse_args()

    root = Path(a.videos).expanduser()
    vids = discover(root)
    if not vids:
        print(f"no video found under {root}")
        return 1
    manifest_levels = levels_from_manifest(root)
    print(f"{len(vids)} videos"
          + (f", {len(manifest_levels)} difficulties from manifest"
             if manifest_levels else ", difficulty inferred from duration"))

    cache_root = Path(a.cache) if a.cache else root / ".ahc_cache"
    enc = features.FrameEncoder(a.encoder, device=a.device, batch_size=32)
    cache = features.Cache(cache_root, enc.model_id)
    print(f"encoder {enc.model_id} on {enc.device}", flush=True)

    # Fixed taxonomy - never read from a label file.
    bank = prompts.build(list(CLASSES))
    bank = features.encode_prompt_bank(enc, bank, cache)

    probe = None
    if a.probe and Path(a.probe).exists():
        z = np.load(a.probe, allow_pickle=True)
        probe = (z["W"], z["b"], [str(c) for c in z["classes"]])
        print(f"D1 probe: {probe[0].shape[0]} classes", flush=True)

    head = None
    if a.head:
        # Fail loudly. A missing checkpoint used to be skipped silently, which
        # sent Difficulty 2 down the onset path meant for long video and cost
        # 9.4 marks before anyone noticed the head was never loaded.
        if not Path(a.head).exists():
            print(f"ERROR: --head {a.head} does not exist")
            return 2
        from ahc.head import TemporalHeadScorer
        head = TemporalHeadScorer(a.head, device=a.device)
        print(f"temporal head: {len(head.classes)} classes from {a.head}", flush=True)
    if a.probe and not Path(a.probe).exists():
        print(f"ERROR: --probe {a.probe} does not exist")
        return 2

    head_ms = None
    if a.head_ms:
        if not Path(a.head_ms).exists():
            print(f"ERROR: --head-ms {a.head_ms} does not exist")
            return 2
        from ahc.head_ms import MSHeadScorer
        head_ms = MSHeadScorer(a.head_ms, device=a.device)
        print(f"multi-scale head: windows {head_ms.cfg.windows} "
              f"({[round(w/a.sample_hz,1) for w in head_ms.cfg.windows]}s)", flush=True)

    def name_interval(default: str, mask) -> str:
        """Class for an interval whose *existence* is already decided.

        Localisation and naming are separate problems and our components are not
        equally good at both. Prompt matching scored 33.3% on class choice
        against the probe's 63.4%, and it is what put
        "stalled_or_broken_down_vehicle" on a dashcam crash compilation.

        But an unconditional override is worse than none. The probe replaced a
        correct "fire" with "loitering" while hedging at 0.30 against a 0.20
        runner-up, and on footage it considers ordinary it names an anomaly
        anyway once "normal" is taken away. So it only speaks when it is
        confident, and silence means the detector keeps its own answer:

          top class is "normal"   the probe sees no anomaly to name and has no
                                  opinion on which one this is. Keep the default.
          margin < MIN_MARGIN     a hedge between two classes is not evidence
                                  against the detector that already fired.

        Confident overrides are the ones worth having: traffic_accident at 0.41
        against a 0.10 runner-up corrects a real mistake, while fire at 0.30
        against 0.20 does not survive the gate and the correct call stands.
        """
        if probe is None or not mask.any():
            return default
        W, bb, pnames = probe
        x = emb[mask].mean(0)
        x = x / (np.linalg.norm(x) + 1e-8)
        lo = W @ x + bb
        e_ = np.exp(lo - lo.max())
        pr = e_ / e_.sum()
        o = np.argsort(-pr)
        top, second = pnames[int(o[0])], float(pr[o[1]])
        if top == "normal" or float(pr[o[0]]) - second < PROBE_MIN_MARGIN:
            return default
        return top

    sub = Submission("run", PUBLIC_MODEL_NAME, f"{enc.device}")
    t_all = time.perf_counter()
    for i, path in enumerate(vids, 1):
        vid = path.stem
        t0 = time.perf_counter()
        # Resolution must match the features the head and probe were fitted on.
        # Feeding 384px frames to a head trained on 256px runs it out of
        # distribution and cost 8.7 marks on Difficulty 2 before this was caught.
        pairs = list(features.sample_frames(path, a.sample_hz, None,
                                            short_side=a.short_side))
        if not pairs:
            sub.add(vid, 1, [], VideoRuntime())
            continue
        times = np.array([t for t, _ in pairs], dtype=np.float32)
        emb = enc.encode_images([f for _, f in pairs])
        duration = float(times[-1]) if len(times) else 0.0
        level = manifest_levels.get(vid) or (
            1 if duration < SHORT_VIDEO_SECONDS
            else 3 if duration > LONG_VIDEO_SECONDS else 2)

        sc = StreamScorer(bank, StreamConfig(sample_hz=a.sample_hz))
        for t, e in zip(times, emb):
            sc.update(float(t), e)
        arr = sc.as_arrays()
        names, cm = sc.class_matrix()

        events: list[PredictedEvent] = []
        if level == 1 and probe is not None:
            # A trained probe beats prompt matching by a wide margin here:
            # 63.4% cross-validated against 33.3% zero-shot on the same clips.
            W, bb, pnames = probe
            x = emb.mean(0)
            x = x / (np.linalg.norm(x) + 1e-8)
            lo = W @ x + bb
            e_ = np.exp(lo - lo.max())
            pr = e_ / e_.sum()
            order = np.argsort(-pr)
            top = pnames[int(order[0])]
            best = next(int(k) for k in order if pnames[int(k)] != "normal")
            if top == "normal" and pr[order[0]] - pr[best] < NORMAL_MIN_MARGIN:
                top = pnames[best]      # too close to call; a guess beats silence
            if top != "normal":
                events.append(PredictedEvent(top, None, None))
        elif level == 1:
            if len(arr["level1"]) and float(arr["level1"].max()) >= D1_THRESHOLD:
                k = int(np.argmax(arr["level1"]))
                events.append(PredictedEvent(names[int(np.argmax(cm[k]))], None, None))
        elif a.d2_cascade and head is not None and level == 2:
            # Two components, two jobs. The head is a reliable video-level
            # detector and a poor localiser: it stays correctly silent on a
            # normal 240s clip, worth a full video's marks, but matched 0 of 12
            # events. The onset path is the reverse - it localises well but has
            # no notion of "nothing here" and fires on the normal video, which
            # costs exactly the marks the head was saving.
            #
            # So the head answers whether, and the onset answers when.
            pa, _ = head.score(emb)
            if float(pa.max()) >= a.d2_hi:
                found = to_events(arr["t"], arr["score"], names, cm,
                                  D3_HI, D3_LO)
                found = sorted(found, key=lambda e: -(e.end - e.start))[:a.d2_cap]
                for e in sorted(found, key=lambda e: e.start):
                    m = (times >= e.start) & (times <= e.end)
                    events.append(PredictedEvent(
                        name_interval(e.class_name, m) if a.name_with_probe
                        else e.class_name, e.start, e.end))
        elif head is not None and level == 2:
            pa, pc = head.score(emb)
            hi_, lo_ = a.d2_hi, a.d2_lo
            if a.d2_adaptive and len(pa) > 8:
                # The head's probability is not comparable across videos: it sits
                # above 0.80 for a whole 240s clip on one and never reaches it on
                # another, so one fixed cutoff either takes everything or nothing.
                # Difficulty 3 does not have this problem because its score is
                # already relative to the video's own baseline. Same idea here -
                # ask which moments are extreme *for this video*.
                hi_ = float(np.quantile(pa, 0.90))
                lo_ = float(np.quantile(pa, 0.75))
                if hi_ - lo_ < 1e-3:        # flat signal, nothing stands out
                    hi_, lo_ = a.d2_hi, a.d2_lo
            tr = HysteresisTracker(hi_, lo_, FAMILIES["default"])
            for t, p in zip(times, pa):
                tr.update(float(t), float(p))
            spans = tr.finish(float(times[-1]))
            if a.refine:
                spans = [refine(times, pa, s, e) for s, e in spans]
            if a.refine and spans:
                pk = lambda sp: float(pa[(times >= sp[0]) & (times <= sp[1])].max()) \
                    if ((times >= sp[0]) & (times <= sp[1])).any() else 0.0
                best = max(pk(sp) for sp in spans)
                spans = [sp for sp in spans if pk(sp) >= PEAK_KEEP_FRACTION * best]
            for s, e in spans:
                m = (times >= s) & (times <= e)
                if m.any():
                    c = head.classes[int(np.argmax(pc[m].mean(axis=0)))]
                    events.append(PredictedEvent(
                        name_interval(c, m) if a.name_with_probe else c, s, e))
        elif a.d3_path == "head" and head is not None:
            # The same head that works on Difficulty 2, just given a longer
            # sequence. Nothing about it is specific to 240s.
            pa, pc = head.score(emb)
            hi_, lo_ = a.d2_hi, a.d2_lo
            if a.d2_adaptive and len(pa) > 8:
                # The head's probability is not comparable across videos: it sits
                # above 0.80 for a whole 240s clip on one and never reaches it on
                # another, so one fixed cutoff either takes everything or nothing.
                # Difficulty 3 does not have this problem because its score is
                # already relative to the video's own baseline. Same idea here -
                # ask which moments are extreme *for this video*.
                hi_ = float(np.quantile(pa, 0.90))
                lo_ = float(np.quantile(pa, 0.75))
                if hi_ - lo_ < 1e-3:        # flat signal, nothing stands out
                    hi_, lo_ = a.d2_hi, a.d2_lo
            tr = HysteresisTracker(hi_, lo_, FAMILIES["default"])
            for t, p in zip(times, pa):
                tr.update(float(t), float(p))
            spans = sorted(tr.finish(float(times[-1])),
                           key=lambda s: -(s[1] - s[0]))[:D3_CAP]
            for s_, e_ in sorted(spans, key=lambda s: s[0]):
                m = (times >= s_) & (times <= e_)
                if m.any():
                    events.append(PredictedEvent(
                        head.classes[int(np.argmax(pc[m].mean(axis=0)))], s_, e_))
        elif a.d3_path == "ms" and head_ms is not None:
            # Long video is what the multi-scale head exists for: its branches
            # span 2s to 60s, where the onset path below has one fixed notion of
            # how long an event lasts.
            pa, pc = head_ms.score(emb)
            tr = HysteresisTracker(0.5, 0.25, FAMILIES["default"])
            for t, p in zip(times, pa):
                tr.update(float(t), float(p))
            spans = tr.finish(float(times[-1]))
            spans = sorted(spans, key=lambda s: -(s[1] - s[0]))[:D3_CAP]
            for s_, e_ in sorted(spans, key=lambda s: s[0]):
                m = (times >= s_) & (times <= e_)
                if m.any():
                    events.append(PredictedEvent(
                        head_ms.classes[int(np.argmax(pc[m].mean(axis=0)))], s_, e_))
        else:
            found = to_events(arr["t"], arr["score"], names, cm, D3_HI, D3_LO)
            sc_t, sc_v = arr["t"], arr["score"]
            peak = lambda ev: float(sc_v[(sc_t >= ev.start) & (sc_t <= ev.end)].max()) \
                if ((sc_t >= ev.start) & (sc_t <= ev.end)).any() else 0.0
            if a.refine and found:
                top = max(peak(e) for e in found)
                found = [e for e in found if peak(e) >= PEAK_KEEP_FRACTION * top]
                found = sorted(found, key=peak, reverse=True)[:a.d3_cap]
            else:
                found = sorted(found, key=lambda e: -(e.end - e.start))[:a.d3_cap]
            for e in sorted(found, key=lambda e: e.start):
                if a.refine:
                    e.start, e.end = refine(sc_t, sc_v, e.start, e.end)
                m = (times >= e.start) & (times <= e.end)
                events.append(PredictedEvent(
                    name_interval(e.class_name, m) if a.name_with_probe
                    else e.class_name, e.start, e.end))

        if a.merge_gap > 0 and level == 3 and len(events) > 1:
            # Difficulty 3 only. Its events run 38-125s, so a union of two
            # fragments still clears tIoU 0.5 - but Difficulty 2 events run ~20s
            # and need a prediction within ~7s of the true start, where widening
            # a span is exactly the wrong move.
            merged = [events[0]]
            for ev in events[1:]:
                prev = merged[-1]
                if (prev.class_name == ev.class_name and prev.end is not None
                        and ev.start is not None
                        and ev.start - prev.end <= a.merge_gap):
                    prev.end = ev.end
                else:
                    merged.append(ev)
            events = merged

        if a.explain:
            from scripts.add_explanations import explain
            for e in events:
                e.explanation = explain(e.class_name, e.start, e.end, None)

        ms = (time.perf_counter() - t0) * 1000
        rt = VideoRuntime(frames_processed=len(times),
                          chunks_processed=max(1, len(times) // 32), end_to_end_ms=ms)
        rt.model("vision-encoder").record(ms / max(len(times), 1))
        sub.add(vid, level, events, rt)
        print(f"[{i}/{len(vids)}] {vid} D{level} {duration:.0f}s "
              f"{len(times)} frames -> {len(events)} events "
              f"({ms:.0f}ms, {duration/max(ms/1000,1e-6):.1f}x realtime)", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sub.as_dict(), indent=2) + "\n")
    el = time.perf_counter() - t_all
    n = sum(len(p["events"]) for p in sub.as_dict()["predictions"])
    print(f"\n{len(vids)} videos, {n} events in {el:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
