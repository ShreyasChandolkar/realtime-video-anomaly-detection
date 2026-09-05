"""Benchmark submission format, and a local scorer that mirrors its rules.

Two things matter about the official scoring that our development metrics were
not measuring:

  - An event at Difficulty 2/3 counts only when the class is right AND
    temporal overlap reaches 0.5. Class-agnostic localisation earns nothing.
  - Difficulty 1 wants a class with null timestamps; Difficulties 2/3 want
    real intervals. Same detector, different rendering.

There is also a latency bonus computed from per-video end_to_end_internal_time_ms,
so timings are recorded as the pipeline runs rather than estimated afterwards.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# The submission taxonomy. "normal" is expressed as an empty events array and
# must never appear as a class name.
CLASSES = [
    "traffic_accident", "traffic_congestion", "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic", "fire", "smoke", "waterlogging_or_flood",
    "wrong_way_driving", "road_spill_or_debris", "fighting_or_violence",
    "loitering_or_suspicious_presence",
]

MARKS = {1: 25.0, 2: 35.0, 3: 40.0}
TIOU_REQUIRED = 0.5


@dataclass
class ModelRuntime:
    model_name: str
    times_ms: list[float] = field(default_factory=list)

    def record(self, ms: float) -> None:
        self.times_ms.append(float(ms))

    def as_dict(self) -> dict:
        t = np.array(self.times_ms) if self.times_ms else np.array([0.0])
        return {
            "model_name": self.model_name,
            "call_count": len(self.times_ms),
            "total_time_ms": round(float(t.sum()), 1),
            "average_time_ms": round(float(t.mean()), 1),
            "p50_time_ms": round(float(np.percentile(t, 50)), 1),
            "p95_time_ms": round(float(np.percentile(t, 95)), 1),
            "max_time_ms": round(float(t.max()), 1),
        }


@dataclass
class VideoRuntime:
    frames_processed: int = 0
    chunks_processed: int = 0
    end_to_end_ms: float = 0.0
    models: dict[str, ModelRuntime] = field(default_factory=dict)

    def model(self, name: str) -> ModelRuntime:
        return self.models.setdefault(name, ModelRuntime(name))

    def as_dict(self) -> dict:
        return {
            "frames_processed": int(self.frames_processed),
            "chunks_processed": int(max(self.chunks_processed, 1)),
            "end_to_end_internal_time_ms": round(float(self.end_to_end_ms), 1),
            "model_runtimes": [m.as_dict() for m in self.models.values()]
                              or [ModelRuntime("vision-encoder").as_dict()],
        }


@dataclass
class PredictedEvent:
    class_name: str
    start: float | None
    end: float | None
    explanation: str = ""

    def as_dict(self, difficulty: int) -> dict:
        d = {"class_name": self.class_name}
        if difficulty == 1:
            d["start_time_sec"] = None
            d["end_time_sec"] = None
        else:
            d["start_time_sec"] = round(float(self.start or 0.0), 3)
            d["end_time_sec"] = round(float(self.end or 0.0), 3)
        if self.explanation and 20 <= len(self.explanation) <= 500:
            d["explanation"] = self.explanation
        return d


class Submission:
    """Accumulates per-video predictions and writes the benchmark JSON."""

    def __init__(self, submission_id: str, model_name: str, hardware: str = "",
                 max_parallel: int = 1):
        self.submission_id = submission_id
        self.model_name = model_name
        self.hardware = hardware
        self.max_parallel = max_parallel
        self.videos: dict[str, dict] = {}
        self._t0 = time.perf_counter()

    def add(self, video_id: str, difficulty: int, events: list[PredictedEvent],
            runtime: VideoRuntime) -> None:
        self.videos[video_id] = {
            "video_id": video_id,
            "events": [e.as_dict(difficulty) for e in events],
            "runtime_metadata": runtime.as_dict(),
        }

    def as_dict(self, manifest: list[str] | None = None) -> dict:
        ids = manifest or sorted(self.videos)
        preds = []
        for vid in ids:
            preds.append(self.videos.get(vid, {
                "video_id": vid, "events": [],
                "runtime_metadata": VideoRuntime().as_dict(),
            }))
        return {
            "schema_version": "1.0",
            "submission_id": self.submission_id,
            "model_name": self.model_name,
            "run_metadata": {
                "total_wall_time_ms": round((time.perf_counter() - self._t0) * 1000, 1),
                "max_parallel_videos": self.max_parallel,
                "hardware": self.hardware,
            },
            "predictions": preds,
        }

    def write(self, path: str | Path, manifest: list[str] | None = None) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(manifest), indent=2) + "\n")
        return p


# ------------------------------------------------------------------ scoring
def _tiou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def score_like_benchmark(submission: dict, gt: pd.DataFrame) -> dict:
    """Local estimate of the official score.

    The exact official formula is not published; this applies the two rules
    that are: Difficulty 1 is a video-level class call, Difficulties 2/3 need
    the right class at tIoU >= 0.5. Treat it as an estimate that tracks the
    real score, not as the real score.
    """
    levels = gt.groupby("video_id")["level"].max().to_dict()
    by_video = {v: sub for v, sub in gt.groupby("video_id")}
    preds = {p["video_id"]: p for p in submission.get("predictions", [])}

    out: dict = {"per_difficulty": {}, "total": 0.0, "max_total": sum(MARKS.values())}
    for diff in (1, 2, 3):
        vids = [v for v, l in levels.items() if l == diff]
        if not vids:
            continue
        if diff == 1:
            correct = 0
            for v in vids:
                truth = set(by_video[v].loc[by_video[v]["class_name"] != "normal",
                                            "class_name"])
                got = {e["class_name"] for e in preds.get(v, {}).get("events", [])}
                # Normal videos: correct exactly when we predicted nothing.
                if not truth:
                    correct += int(not got)
                else:
                    correct += int(bool(truth & got))
            frac = correct / len(vids)
            out["per_difficulty"][diff] = {
                "videos": len(vids), "correct": correct,
                "fraction": round(frac, 3),
                "marks": round(frac * MARKS[diff], 2), "max": MARKS[diff],
            }
        else:
            tp = fp = fn = 0
            for v in vids:
                truths = [(float(r["start_time_sec"]), float(r["end_time_sec"]),
                           str(r["class_name"]))
                          for _, r in by_video[v].iterrows()
                          if r["class_name"] != "normal"
                          and pd.notna(r["start_time_sec"])]
                got = preds.get(v, {}).get("events", [])
                used = set()
                for e in got:
                    best, best_i = None, 0.0
                    for k, (s, en, c) in enumerate(truths):
                        if k in used or c != e["class_name"]:
                            continue
                        i = _tiou((float(e.get("start_time_sec") or 0),
                                   float(e.get("end_time_sec") or 0)), (s, en))
                        if i > best_i:
                            best, best_i = k, i
                    if best is not None and best_i >= TIOU_REQUIRED:
                        used.add(best)
                        tp += 1
                    else:
                        fp += 1
                fn += len(truths) - len(used)
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            out["per_difficulty"][diff] = {
                "videos": len(vids), "tp": tp, "fp": fp, "fn": fn,
                "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
                "marks": round(f1 * MARKS[diff], 2), "max": MARKS[diff],
            }
        out["total"] += out["per_difficulty"][diff]["marks"]
    out["total"] = round(out["total"], 2)
    return out


def render_score(s: dict) -> str:
    lines = [f"estimated score: {s['total']:.1f} / {s['max_total']:.0f}"]
    for diff, d in sorted(s["per_difficulty"].items()):
        if diff == 1:
            lines.append(f"  D1  {d['correct']}/{d['videos']} videos correct"
                         f"   {d['marks']:.1f}/{d['max']:.0f}")
        else:
            lines.append(f"  D{diff}  P {d['precision']:.2f} R {d['recall']:.2f} "
                         f"F1 {d['f1']:.2f}  (tp{d['tp']} fp{d['fp']} fn{d['fn']})"
                         f"   {d['marks']:.1f}/{d['max']:.0f}")
    return "\n".join(lines)
