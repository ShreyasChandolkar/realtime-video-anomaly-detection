"""Score curve -> event intervals.

Hysteresis with per-family dwell requirements. The families exist because the
events genuinely differ in time: a collision is over in about a second, a queue
builds over a minute, a stalled vehicle is only an event once it has been still
for a while. A single global threshold cannot express that.

Family assignment falls back to a neutral default, so an unseen class still
produces intervals without a code change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Family:
    min_on_s: float      # evidence must persist this long before an event opens
    min_off_s: float     # ...and fall away this long before it closes
    merge_gap_s: float   # neighbouring events closer than this become one


FAMILIES: dict[str, Family] = {
    "instant":    Family(min_on_s=0.5, min_off_s=1.5, merge_gap_s=2.0),
    "gradual":    Family(min_on_s=4.0, min_off_s=6.0, merge_gap_s=8.0),
    "persistent": Family(min_on_s=6.0, min_off_s=4.0, merge_gap_s=6.0),
    "default":    Family(min_on_s=2.0, min_off_s=3.0, merge_gap_s=4.0),
}

CLASS_FAMILY: dict[str, str] = {
    "traffic_accident": "instant",
    "fighting_or_violence": "instant",
    "traffic_congestion": "gradual",
    "waterlogging_or_flood": "gradual",
    "smoke": "gradual",
    "fire": "gradual",
    "stalled_or_broken_down_vehicle": "persistent",
    "vehicle_blocking_traffic": "persistent",
    "loitering_or_suspicious_presence": "persistent",
    "road_spill_or_debris": "persistent",
    "wrong_way_driving": "default",
}


def family_for(class_name: str | None) -> Family:
    return FAMILIES[CLASS_FAMILY.get(class_name or "", "default")]


@dataclass
class Event:
    start: float
    end: float
    class_name: str
    score: float

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


class HysteresisTracker:
    """Online two-threshold segmentation, one sample at a time.

    The live path and the offline evaluation both drive this object, so there is
    exactly one implementation of when an event opens and closes. A causality
    bug therefore cannot hide in one path and not the other.
    """

    def __init__(self, hi: float, lo: float, family: Family):
        self.hi, self.lo, self.family = hi, lo, family
        self.on = False
        self.start: float | None = None
        self._cand_start: float | None = None
        self._cand_end: float | None = None
        self.closed: list[tuple[float, float]] = []

    def update(self, t: float, score: float) -> tuple[float, float] | None:
        """Feed one sample. Returns a span if an event just closed."""
        if not self.on:
            if score >= self.hi:
                if self._cand_start is None:
                    self._cand_start = t
                elif t - self._cand_start >= self.family.min_on_s:
                    self.on, self.start = True, self._cand_start
                    self._cand_start = self._cand_end = None
            else:
                self._cand_start = None
            return None

        if score < self.lo:
            if self._cand_end is None:
                self._cand_end = t
            elif t - self._cand_end >= self.family.min_off_s:
                span = (self.start, self._cand_end)
                self.closed.append(span)
                self.on, self._cand_start, self._cand_end = False, None, None
                self.start = None
                return span
        else:
            self._cand_end = None
        return None

    def finish(self, t_last: float) -> list[tuple[float, float]]:
        """Close any open event and return every span, gap-merged."""
        spans = list(self.closed)
        if self.on and self.start is not None:
            spans.append((self.start, float(t_last)))
        merged: list[tuple[float, float]] = []
        for s, e in spans:
            if merged and s - merged[-1][1] <= self.family.merge_gap_s:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        return merged


def hysteresis(t: np.ndarray, score: np.ndarray, hi: float, lo: float,
               family: Family) -> list[tuple[float, float]]:
    """Batch wrapper over the online tracker - same logic, replayed."""
    if len(t) == 0:
        return []
    tr = HysteresisTracker(hi, lo, family)
    for ti, si in zip(t, score):
        tr.update(float(ti), float(si))
    return tr.finish(float(t[-1]))


def to_events(t: np.ndarray, score: np.ndarray, classes: list[str],
              class_scores: np.ndarray, hi: float, lo: float,
              default_family: str = "default") -> list[Event]:
    """Segment, then name each span by its dominant class.

    Two passes: a neutral segmentation to find *where* something happens, then a
    class-aware re-segmentation using that class's own temporal family. This
    matters because the right dwell time is not knowable until you know what the
    event is.
    """
    if len(t) == 0:
        return []
    coarse = hysteresis(t, score, hi, lo, FAMILIES[default_family])
    events: list[Event] = []
    for s, e in coarse:
        m = (t >= s) & (t <= e)
        if not m.any():
            continue
        mean_cls = class_scores[m].mean(axis=0)
        cls = classes[int(np.argmax(mean_cls))] if len(classes) else "normal"
        fam = family_for(cls)
        for s2, e2 in hysteresis(t[m], score[m], hi, lo, fam) or [(s, e)]:
            m2 = (t >= s2) & (t <= e2)
            events.append(Event(float(s2), float(e2), cls,
                                float(score[m2].mean()) if m2.any() else float(score[m].mean())))
    return events


def calibrate_families(gt) -> dict[str, str]:
    """Assign families from observed event durations rather than intuition."""
    out = dict(CLASS_FAMILY)
    if gt is None or len(gt) == 0:
        return out
    sub = gt[(gt["class_name"] != "normal") & gt["start_time_sec"].notna()
             & gt["end_time_sec"].notna()].copy()
    if not len(sub):
        return out
    sub["dur"] = sub["end_time_sec"] - sub["start_time_sec"]
    for cls, grp in sub.groupby("class_name"):
        med = float(grp["dur"].median())
        out[str(cls)] = ("instant" if med < 3.0
                         else "default" if med < 10.0
                         else "gradual" if med < 30.0
                         else "persistent")
    return out
