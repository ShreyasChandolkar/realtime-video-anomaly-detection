"""Causal streaming anomaly scorer.

Strictly online: every value emitted at time t is a function of samples at
times <= t. There is no global normalisation over a clip, no lookahead, and no
second pass. The offline evaluation replays recorded features through this same
object with a simulated clock, so the number we tune against is produced by the
code that runs live.

Two complementary signals, deliberately chosen to cover each other's blind spot:

  deviation  relative, self-referential. How far is now from what this scene has
             been doing? Open-set by construction — it needs no name for the
             event — but it requires a stretch of routine to learn from, so it
             is blind to a clip that is anomalous from its first frame.

  semantic   absolute, language-anchored. A log-ratio of similarity to anomaly
             prompts against normal prompts. Needs the event to be nameable (or
             at least near something nameable) but works from frame one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .prompts import NORMAL, PromptBank


def _alpha(dt: float, halflife_s: float) -> float:
    """dt-aware EWMA weight, so the model is frame-rate independent."""
    if halflife_s <= 0:
        return 1.0
    return 1.0 - math.pow(0.5, max(dt, 1e-6) / halflife_s)


class Ewma:
    """Scalar exponentially-weighted mean and variance."""

    __slots__ = ("halflife", "mean", "var", "n")

    def __init__(self, halflife_s: float):
        self.halflife = halflife_s
        self.mean = 0.0
        self.var = 0.0
        self.n = 0

    def update(self, x: float, dt: float) -> None:
        a = _alpha(dt, self.halflife)
        if self.n == 0:
            self.mean, self.var = float(x), 0.0
        else:
            d = float(x) - self.mean
            self.mean += a * d
            self.var = (1 - a) * (self.var + a * d * d)
        self.n += 1

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 0.0))

    def z(self, x: float, floor: float = 1e-3) -> float:
        if self.n < 2:
            return 0.0
        return (float(x) - self.mean) / max(self.std, floor)


@dataclass
class StreamConfig:
    sample_hz: float = 4.0
    # deviation path
    baseline_halflife_s: float = 20.0   # how fast "normal" for this scene adapts
    warmup_s: float = 4.0               # emit no deviation before this
    znorm_halflife_s: float = 45.0      # self-normalisation of the deviation signal
    # semantic path
    # Observed per-frame cosine spread on this corpus is ~0.14, so a temperature
    # near 0.02 keeps the class softmax informative instead of near-uniform.
    temperature: float = 0.02
    # How a class's several phrasings collapse into one score.
    #
    # "max" takes the best-matching phrasing, which is what created the count
    # bias: the maximum of many draws beats the maximum of few regardless of
    # content, so a class with more phrasings wins on arithmetic. "mean" is the
    # standard CLIP prompt-ensembling recipe - average the phrasing embeddings
    # into one class vector - and is invariant to that count.
    #
    # Measured on 223 balanced train clips over 6 classes, "mean" won on both:
    #   max    anomaly-vs-normal 63.7%   class top-1 29.1%   top-3 64.8%
    #   mean   anomaly-vs-normal 70.9%   class top-1 31.8%   top-3 66.5%
    #
    # And then lost 24 marks on the benchmark: 51.9 -> 27.5. The default stays
    # "max" because that is what scored 51.9.
    #
    # The gap is the lesson. Train clips are short and single-class, so train
    # measures "name this clip". The benchmark measures intervals at tIoU >= 0.5
    # with the right class. Changing the pooling shifts the score's scale *and*
    # its dynamics, which silently invalidates thresholds fitted to the old
    # curve - matching event counts, which was done, is not enough. Any future
    # change here has to refit hi/lo/level1 against interval matching on long
    # multi-event footage, which no train clip provides.
    class_pooling: str = "max"
    # Cosine margins run about -0.05..+0.14, so lift them onto a scale where a
    # threshold near 1.0 is meaningful alongside the z-scored deviation.
    semantic_gain: float = 20.0
    # fusion
    #
    # PROVISIONAL. The structural choices below are mechanism-driven and hold
    # regardless of dataset, but the numbers quoted came from the public test
    # set before that set was quarantined. They are recorded as the reason for
    # the design, not as evidence of generalisation, and are due to be
    # re-derived on train. See scripts/calibrate.py.
    #
    # Video-level AUC when measured:
    #   gated       0.911      deviation amplifies semantic evidence
    #   semantic    0.893      language alone
    #   additive    0.726      the two summed
    #   deviation   0.619      deviation alone, near chance
    #
    # Deviation on its own answers "did something change", which a pan, a light
    # change or a passing truck all satisfy. Summing it lets those raise an
    # alarm unaided, which is why the additive form scored below language alone.
    # Gating keeps it as a multiplier on evidence the semantics already support.
    # Absolute semantic level answers "is anything wrong in this video" and
    # scores 0.89-0.92 video AUC. It cannot answer "when did it start": where a
    # whole clip looks like the event the level never falls and the interval
    # never closes, which put predicted coverage at 61% of every video and event
    # F1 at 0.007. Subtracting the video's own slow semantic baseline gives an
    # onset signal that lifts event F1@0.3 to 0.400 at 7% coverage, while being
    # useless video-level (0.577) for the same reason inverted.
    #
    # So the two questions get two readouts from one scorer rather than one
    # compromised number.
    onset_halflife_s: float = 120.0
    w_onset: float = 1.0         # 1.0 = intervals driven purely by onset
    # Thresholds live in configs/calibration.json, fitted by scripts/calibrate.py
    # on train. Anything hardcoded here is a starting point, not a result.
    fusion: str = "gated"        # "gated" | "additive" | "semantic"
    w_deviation: float = 0.45    # additive mode only
    w_semantic: float = 0.55     # additive mode only
    gate_gain: float = 0.5       # how much deviation can amplify, gated mode
    gate_centre: float = 2.0     # deviation z at which the gate half-opens
    # A near-static scene drives the running variance toward zero, which makes
    # the z-score explode on the first trivial change. Both guards below are
    # about false alarms on quiet cameras, which are scored as harshly as misses.
    dev_clip: float = 8.0        # cap on the deviation z-score
    dev_rel_floor: float = 0.15  # std floor, relative to mean deviation
    # kinematics (optional, may be absent)
    w_kinematic: float = 0.0


@dataclass
class ScoreFrame:
    t: float
    score: float                 # fused anomaly score
    deviation: float             # z-scored, 0 during warmup
    semantic: float              # absolute, language-anchored
    onset: float                 # semantic minus this video's own slow baseline
    level1: float                # video-level readout (absolute)
    warm: bool
    class_scores: dict[str, float] = field(default_factory=dict)

    @property
    def top_class(self) -> str | None:
        if not self.class_scores:
            return None
        return max(self.class_scores, key=self.class_scores.get)


class StreamScorer:
    """One instance per stream. Reset between videos."""

    def __init__(self, bank: PromptBank, config: StreamConfig | None = None):
        if bank.embeddings is None:
            raise ValueError("PromptBank must be encoded before scoring")
        self.bank = bank
        self.cfg = config or StreamConfig()
        self._normal = bank.normal_mask
        self._anom = ~self._normal
        self._class_idx = {c: np.where(bank.class_mask(c))[0] for c in bank.classes}
        self._centroids = self._build_centroids()
        self.reset()

    def _build_centroids(self) -> dict[str, np.ndarray]:
        """One unit vector per class (and normal), averaged over its phrasings."""
        out = {}
        for name, idx in list(self._class_idx.items()) + [
                ("__normal__", np.where(self._normal)[0])]:
            if not len(idx):
                continue
            v = self.bank.embeddings[idx].mean(axis=0)
            out[name] = v / (np.linalg.norm(v) + 1e-8)
        return out

    def reset(self) -> None:
        self._mu: np.ndarray | None = None
        self._t0: float | None = None
        self._t_prev: float | None = None
        self._dev_stats = Ewma(self.cfg.znorm_halflife_s)
        self._sem_base = Ewma(self.cfg.onset_halflife_s)
        self.history: list[ScoreFrame] = []

    # -- signals -----------------------------------------------------------
    def _semantic(self, e: np.ndarray) -> tuple[float, dict[str, float]]:
        """Best-anomaly minus best-normal similarity, plus the class split.

        Deliberately a *margin between maxima*, not a ratio of sums. Summing over
        prompt groups makes the score depend on how many phrasings each group
        happens to contain - with ~100 anomaly prompts against 6 normal ones,
        every frame scores as anomalous regardless of content. Taking the best
        phrasing from each side makes the comparison independent of bank size,
        so adding prompts can sharpen the score but cannot inflate it.
        """
        if self.cfg.class_pooling == "mean":
            per_class = {c: float(np.dot(self._centroids[c], e))
                         for c in self._class_idx if c in self._centroids}
            best_normal = (float(np.dot(self._centroids["__normal__"], e))
                           if "__normal__" in self._centroids else 0.0)
        else:
            sims = self.bank.embeddings @ e        # (P,) cosine, both normalised
            best_normal = float(sims[self._normal].max()) if self._normal.any() else 0.0
            per_class = {c: float(sims[idx].max()) for c, idx in self._class_idx.items()}
        best_anom = max(per_class.values()) if per_class else 0.0
        score = (best_anom - best_normal) * self.cfg.semantic_gain

        # Softmax over per-class maxima, so class probabilities reflect the best
        # phrasing per class rather than how many phrasings it was given.
        if per_class:
            names = list(per_class)
            v = np.array([per_class[c] for c in names], dtype=np.float32)
            v = (v - v.max()) / max(self.cfg.temperature, 1e-6)
            p = np.exp(v)
            p /= p.sum() + 1e-12
            classes = {c: float(pi) for c, pi in zip(names, p)}
        else:
            classes = {}
        return score, classes

    def _deviation(self, e: np.ndarray, dt: float, elapsed: float) -> float:
        if self._mu is None:
            self._mu = e.copy()
            return 0.0
        mu_hat = self._mu / (np.linalg.norm(self._mu) + 1e-8)
        raw = 1.0 - float(np.dot(e, mu_hat))       # cosine distance to own baseline
        a = _alpha(dt, self.cfg.baseline_halflife_s)
        self._mu += a * (e - self._mu)
        if elapsed < self.cfg.warmup_s:
            self._dev_stats.update(raw, dt)
            return 0.0
        floor = max(1e-3, self.cfg.dev_rel_floor * abs(self._dev_stats.mean))
        z = self._dev_stats.z(raw, floor=floor)
        self._dev_stats.update(raw, dt)
        return float(np.clip(z, -self.cfg.dev_clip, self.cfg.dev_clip))

    def _gate(self, deviation: float) -> float:
        return 1.0 / (1.0 + math.exp(-(deviation - self.cfg.gate_centre)))

    def _fuse(self, base: float, deviation: float) -> float:
        """Combine with deviation. See StreamConfig.fusion for the measurements."""
        if self.cfg.fusion == "semantic":
            return base
        if self.cfg.fusion == "additive":
            return self.cfg.w_semantic * base + self.cfg.w_deviation * deviation
        return base * (1.0 + self.cfg.gate_gain * self._gate(deviation))

    # -- main entry point --------------------------------------------------
    def update(self, t_sec: float, embedding: np.ndarray,
               kinematic: float | None = None) -> ScoreFrame:
        e = np.asarray(embedding, dtype=np.float32)
        e = e / (np.linalg.norm(e) + 1e-8)

        if self._t0 is None:
            self._t0 = t_sec
        dt = 1.0 / self.cfg.sample_hz if self._t_prev is None else max(t_sec - self._t_prev, 1e-3)
        self._t_prev = t_sec
        elapsed = t_sec - self._t0

        semantic, classes = self._semantic(e)
        deviation = self._deviation(e, dt, elapsed)
        warm = elapsed >= self.cfg.warmup_s

        # Onset: how far above this video's own slow semantic baseline are we.
        onset = semantic - self._sem_base.mean if self._sem_base.n else 0.0
        self._sem_base.update(semantic, dt)

        w = self.cfg.w_onset
        score = self._fuse((1.0 - w) * semantic + w * onset, deviation)
        level1 = self._fuse(semantic, deviation)
        if kinematic is not None and self.cfg.w_kinematic:
            score += self.cfg.w_kinematic * float(kinematic)

        frame = ScoreFrame(t=float(t_sec), score=float(score),
                           deviation=float(deviation), semantic=float(semantic),
                           onset=float(onset), level1=float(level1),
                           warm=bool(warm), class_scores=classes)
        self.history.append(frame)
        return frame

    # -- convenience -------------------------------------------------------
    def as_arrays(self) -> dict[str, np.ndarray]:
        if not self.history:
            return {k: np.zeros(0, dtype=np.float32)
                    for k in ("t", "score", "deviation", "semantic", "onset", "level1")}
        return {
            "t": np.array([f.t for f in self.history], dtype=np.float32),
            "score": np.array([f.score for f in self.history], dtype=np.float32),
            "deviation": np.array([f.deviation for f in self.history], dtype=np.float32),
            "semantic": np.array([f.semantic for f in self.history], dtype=np.float32),
            "onset": np.array([f.onset for f in self.history], dtype=np.float32),
            "level1": np.array([f.level1 for f in self.history], dtype=np.float32),
        }

    def class_matrix(self) -> tuple[list[str], np.ndarray]:
        classes = list(self.bank.classes)
        if not self.history:
            return classes, np.zeros((0, len(classes)), dtype=np.float32)
        m = np.array([[f.class_scores.get(c, 0.0) for c in classes]
                      for f in self.history], dtype=np.float32)
        return classes, m
