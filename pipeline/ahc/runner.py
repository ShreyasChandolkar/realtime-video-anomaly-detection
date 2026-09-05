"""Replay cached features through the live scorer.

The offline evaluation and the live demo run the *same* StreamScorer. The only
difference is the clock: here timestamps come from the cache and are consumed as
fast as the CPU allows; live they arrive in wall time. That equivalence is the
point - a causality bug shows up in the score rather than only in the demo.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .align import apply_rotation
from .features import Cache
from .postproc import Event, to_events
from .stream import ScoreFrame, StreamConfig, StreamScorer


@dataclass
class RunConfig:
    # Calibrated on the public test set: hi=0.4 / lo=0.16 maximised event
    # F1@0.3 (0.417) without costing video-level AUC (0.911).
    rotation: object = None  # optional frame->prompt alignment, see ahc/align.py
    hi: float = 0.4          # enter threshold on the fused score
    lo: float = 0.16         # exit threshold (hysteresis)
    # Level 1 is a separate decision from "did an interval open". Deriving the
    # binary label from interval existence gave F1 0.432 despite AUC 0.911,
    # because the onset signal that localises well stays quiet on videos that
    # are anomalous throughout. Thresholding the absolute readout instead:
    # P 0.931 R 0.964 F1 0.947 acc 0.912 on public test.
    level1_threshold: float = 0.40
    stream: StreamConfig = None

    def __post_init__(self):
        if self.stream is None:
            self.stream = StreamConfig()


@dataclass
class VideoResult:
    video_id: str
    split: str
    t: np.ndarray
    score: np.ndarray
    deviation: np.ndarray
    semantic: np.ndarray
    onset: np.ndarray
    level1: np.ndarray
    classes: list[str]
    class_scores: np.ndarray
    events: list[Event]

    level1_threshold: float = 0.40

    @property
    def is_anomaly(self) -> int:
        """Video-level call from the absolute score, independent of intervals."""
        return int(self.video_score >= self.level1_threshold)

    @property
    def top_event(self) -> Event | None:
        return max(self.events, key=lambda e: e.score) if self.events else None

    @property
    def video_score(self) -> float:
        """Level 1 uses the absolute readout - see StreamConfig.fusion."""
        v = self.level1 if len(self.level1) else self.score
        return float(v.max()) if len(v) else float("-inf")

    def to_row(self) -> dict:
        top = self.top_event
        return {
            "video_id": self.video_id, "split": self.split,
            "score": self.video_score, "is_anomaly": self.is_anomaly,
            "class_name": top.class_name if top else "normal",
            "start_time_sec": top.start if top else None,
            "end_time_sec": top.end if top else None,
            "n_events": len(self.events),
        }


def run_video(bank, cache: Cache, split: str, video_id: str,
              cfg: RunConfig | None = None,
              realtime: bool = False,
              on_frame=None) -> VideoResult:
    """Stream one video's cached embeddings through the scorer."""
    cfg = cfg or RunConfig()
    emb, meta = cache.load(split, video_id)
    times = np.asarray(meta.get("timestamps", []), dtype=np.float32)
    emb = np.asarray(emb, dtype=np.float32)
    if cfg.rotation is not None:
        emb = apply_rotation(emb, cfg.rotation)
    if len(times) != len(emb):
        n = min(len(times), len(emb))
        times, emb = times[:n], emb[:n]

    sc = StreamScorer(bank, cfg.stream)
    wall0 = time.perf_counter()
    for ti, ei in zip(times, emb):
        if realtime:
            due = wall0 + float(ti)
            now = time.perf_counter()
            if due > now:
                time.sleep(min(due - now, 1.0))
        frame = sc.update(float(ti), ei)
        if on_frame is not None:
            on_frame(frame)

    arr = sc.as_arrays()
    names, cm = sc.class_matrix()
    events = to_events(arr["t"], arr["score"], names, cm, cfg.hi, cfg.lo)
    return VideoResult(video_id=video_id, split=split, t=arr["t"], score=arr["score"],
                       deviation=arr["deviation"], semantic=arr["semantic"],
                       onset=arr["onset"], level1=arr["level1"],
                       classes=names, class_scores=cm, events=events)


def run_split(bank, cache: Cache, videos: list[tuple[str, str]],
              cfg: RunConfig | None = None, verbose: bool = False):
    """videos: [(split, video_id), ...]. Returns (rows_df, curves, events)."""
    cfg = cfg or RunConfig()
    rows, curves, events = [], {}, {}
    for split, vid in videos:
        if not cache.has(split, vid):
            continue
        try:
            r = run_video(bank, cache, split, vid, cfg)
            r.level1_threshold = cfg.level1_threshold
        except Exception as e:
            if verbose:
                print(f"  {vid}: {type(e).__name__} {e}")
            continue
        rows.append(r.to_row())
        curves[vid] = (r.t, r.score)
        events[vid] = r.events
        if verbose:
            print(f"  {vid:14s} {len(r.events)} ev  max {r.video_score:6.2f}  "
                  f"{r.top_event.class_name if r.top_event else '-'}")
    return pd.DataFrame(rows), curves, events


def sweep_thresholds(bank, cache: Cache, videos, gt,
                     his=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
                     ratio: float = 0.4,
                     stream_cfg: StreamConfig | None = None) -> pd.DataFrame:
    """Threshold sweep on cached features.

    Cheap enough to run exhaustively because nothing here touches a GPU or a
    video file - this is the payoff of the feature cache.
    """
    from . import metrics
    out = []
    for hi in his:
        cfg = RunConfig(hi=hi, lo=hi * ratio, stream=stream_cfg)
        preds, curves, events = run_split(bank, cache, videos, cfg)
        if not len(preds):
            continue
        s = metrics.summarise(preds, curves, events, gt)
        out.append({
            "hi": hi, "lo": hi * ratio,
            "auc": s["level1"].get("auc"),
            "bin_f1": s["level1"].get("binary_f1"),
            "cls_acc": s["level1"].get("class_accuracy"),
            "frame_auc": s["frame"].get("frame_auc"),
            "ev_f1@0.3": s["event"]["tiou@0.3"]["class_aware"]["f1"],
            "ev_f1@0.3_agn": s["event"]["tiou@0.3"]["class_agnostic"]["f1"],
        })
    return pd.DataFrame(out)
