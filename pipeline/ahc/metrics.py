"""Scoring. Deliberately a suite, not a single number.

The official private metric is not published, so we compute the family it is
almost certainly drawn from and watch all of them. The one to trust for
generalisation is always the public-test figure, never a train holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NORMAL = "normal"


# -- primitives ------------------------------------------------------------
def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based AUC. No sklearn dependency, handles ties."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos, neg = int(y_true.sum()), int((1 - y_true).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y_true == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], a[0]) - a[0] + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


# -- ground-truth shaping --------------------------------------------------
def gt_intervals(gt: pd.DataFrame, video_id: str) -> list[tuple[float, float, str]]:
    sub = gt[(gt["video_id"] == video_id) & (gt["class_name"] != NORMAL)]
    out = []
    for _, r in sub.iterrows():
        s, e = r.get("start_time_sec"), r.get("end_time_sec")
        if pd.isna(s) or pd.isna(e):
            continue
        out.append((float(s), float(e), str(r["class_name"])))
    return out


def frame_labels(intervals, t: np.ndarray) -> np.ndarray:
    y = np.zeros(len(t), dtype=int)
    for s, e, _ in intervals:
        y[(t >= s) & (t <= e)] = 1
    return y


# -- level 1: video level --------------------------------------------------
def video_level(pred: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """pred: video_id, score, is_anomaly, class_name (one row per video)."""
    g = (gt.groupby("video_id")
           .agg(is_anomaly=("is_anomaly", "max"),
                classes=("class_name", lambda s: sorted(set(s) - {NORMAL})))
           .reset_index())
    m = pred.merge(g, on="video_id", how="inner", suffixes=("_pred", "_gt"))
    if not len(m):
        return {"n": 0}
    y = m["is_anomaly_gt"].values.astype(int)
    out = {"n": int(len(m)), "n_anomalous": int(y.sum())}
    if "score" in m.columns:
        out["auc"] = auc_roc(y, m["score"].values)
    if "is_anomaly_pred" in m.columns:
        yp = m["is_anomaly_pred"].values.astype(int)
        out |= {f"binary_{k}": v for k, v in
                prf(int(((yp == 1) & (y == 1)).sum()),
                    int(((yp == 1) & (y == 0)).sum()),
                    int(((yp == 0) & (y == 1)).sum())).items()}
        out["binary_accuracy"] = float((yp == y).mean())
    if "class_name" in m.columns:
        anom = m[y == 1]
        if len(anom):
            hit = [row["class_name"] in row["classes"] for _, row in anom.iterrows()]
            out["class_accuracy"] = float(np.mean(hit))
    return out


# -- levels 2/3: temporal --------------------------------------------------
def frame_level(curves: dict[str, tuple[np.ndarray, np.ndarray]],
                gt: pd.DataFrame) -> dict:
    """curves: video_id -> (t, score). Pooled frame-level AUC, the standard
    VAD figure — comparable to published numbers."""
    ys, ss = [], []
    for vid, (t, s) in curves.items():
        if len(t) == 0:
            continue
        ys.append(frame_labels(gt_intervals(gt, vid), t))
        ss.append(np.asarray(s))
    if not ys:
        return {"frame_auc": float("nan"), "n_frames": 0}
    y, s = np.concatenate(ys), np.concatenate(ss)
    return {"frame_auc": auc_roc(y, s), "n_frames": int(len(y)),
            "anomalous_frac": float(y.mean())}


def event_level(pred_events: dict[str, list], gt: pd.DataFrame,
                tiou: float = 0.3, require_class: bool = True) -> dict:
    """Greedy one-to-one matching at a tIoU threshold."""
    tp = fp = fn = 0
    for vid in set(pred_events) | set(gt["video_id"]):
        preds = sorted(pred_events.get(vid, []), key=lambda e: -e.score)
        truths = gt_intervals(gt, vid)
        used = set()
        for p in preds:
            best, best_iou = None, 0.0
            for k, (s, e, c) in enumerate(truths):
                if k in used or (require_class and c != p.class_name):
                    continue
                i = temporal_iou((p.start, p.end), (s, e))
                if i > best_iou:
                    best, best_iou = k, i
            if best is not None and best_iou >= tiou:
                used.add(best)
                tp += 1
            else:
                fp += 1
        fn += len(truths) - len(used)
    return prf(tp, fp, fn)


def summarise(pred_videos, curves, pred_events, gt,
              tious=(0.1, 0.3, 0.5)) -> dict:
    out = {"level1": video_level(pred_videos, gt), "frame": frame_level(curves, gt)}
    out["event"] = {
        f"tiou@{t}": {
            "class_aware": event_level(pred_events, gt, t, True),
            "class_agnostic": event_level(pred_events, gt, t, False),
        } for t in tious
    }
    return out


def render(summary: dict) -> str:
    l1, fr = summary.get("level1", {}), summary.get("frame", {})
    lines = [
        f"videos {l1.get('n', 0)} ({l1.get('n_anomalous', 0)} anomalous)",
        f"  L1  auc {l1.get('auc', float('nan')):.3f}"
        f"  binF1 {l1.get('binary_f1', float('nan')):.3f}"
        f"  clsAcc {l1.get('class_accuracy', float('nan')):.3f}",
        f"  frame auc {fr.get('frame_auc', float('nan')):.3f}"
        f"  ({fr.get('n_frames', 0)} samples, {fr.get('anomalous_frac', 0):.1%} positive)",
    ]
    for k, v in summary.get("event", {}).items():
        lines.append(f"  {k:10s} class-aware F1 {v['class_aware']['f1']:.3f}"
                     f"   agnostic F1 {v['class_agnostic']['f1']:.3f}")
    return "\n".join(lines)
