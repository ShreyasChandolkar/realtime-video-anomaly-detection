"""Live streaming worker: video in, annotated frames and anomaly state out.

Paced to wall clock, so what you watch is what a real deployment would see. The
scorer and the hysteresis tracker are the same objects the offline evaluation
drives - only the clock differs.

Runs comfortably on an M-series laptop: SigLIP2-base at 4 Hz costs about 16 ms
per sample against a 250 ms budget, so the display never waits on the model.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .postproc import FAMILIES, HysteresisTracker, family_for
from .stream import StreamConfig, StreamScorer


@dataclass
class LiveConfig:
    sample_hz: float = 4.0
    hi: float = 0.4
    lo: float = 0.16
    level1_threshold: float = 0.40   # video-level call, independent of intervals
    speed: float = 1.0          # 1.0 = real time; 0 = as fast as possible
    loop: bool = True
    short_side: int = 256       # what the encoder sees
    display_width: int = 960    # what the browser sees
    history: int = 600          # samples kept for the curve (~2.5 min at 4 Hz)
    stream: StreamConfig = field(default_factory=StreamConfig)


@dataclass
class LiveEvent:
    class_name: str
    start: float
    end: float | None
    peak: float
    last_t: float = 0.0     # latest sample seen while this event is open

    def as_dict(self) -> dict:
        # A live event has no end yet, so its duration runs to the current
        # sample. Using `end or start` reported 0s for everything in progress.
        until = self.end if self.end is not None else max(self.last_t, self.start)
        return {"class_name": self.class_name, "start": round(self.start, 2),
                "end": round(self.end, 2) if self.end is not None else None,
                "peak": round(self.peak, 2),
                "duration": round(until - self.start, 2)}


class LiveWorker(threading.Thread):
    """Decode -> embed -> score -> annotate, paced to wall clock."""

    daemon = True

    def __init__(self, video_path: str, encoder, bank, config: LiveConfig | None = None):
        super().__init__(name="live-worker")
        self.video_path = video_path
        self.encoder = encoder
        self.bank = bank
        self.cfg = config or LiveConfig()

        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._state: dict = {"ready": False}
        self._stop = threading.Event()

        self.scorer = StreamScorer(bank, self.cfg.stream)
        self.tracker = HysteresisTracker(self.cfg.hi, self.cfg.lo, FAMILIES["default"])
        self.events: list[LiveEvent] = []
        self.active: LiveEvent | None = None
        self._active_cls_sum: np.ndarray | None = None
        self._active_n = 0
        self.curve: deque = deque(maxlen=self.cfg.history)
        self.level1_max: float = float("-inf")
        self.level1_cls: str | None = None

    # ------------------------------------------------------------------ api
    def stop(self) -> None:
        self._stop.set()

    @property
    def jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    @property
    def state(self) -> dict:
        with self._lock:
            return dict(self._state)

    # --------------------------------------------------------------- render
    def _annotate(self, frame, t_sec: float, score: float, label: str | None):
        import cv2

        h, w = frame.shape[:2]
        if w > self.cfg.display_width:
            s = self.cfg.display_width / w
            frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]

        bar_h = 34
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        norm = float(np.clip((score - self.cfg.lo) / max(self.cfg.hi * 2 - self.cfg.lo, 1e-6),
                             0.0, 1.0))
        colour = ((60, 200, 90) if score < self.cfg.lo
                  else (40, 190, 240) if score < self.cfg.hi
                  else (40, 60, 240))
        cv2.rectangle(frame, (8, 10), (8 + int(220 * norm), 24), colour, -1)
        cv2.rectangle(frame, (8, 10), (228, 24), (120, 120, 130), 1)
        cv2.putText(frame, f"{t_sec:6.1f}s  score {score:+5.2f}", (240, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 240), 1, cv2.LINE_AA)

        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            cv2.rectangle(frame, (8, bar_h + 8), (8 + tw + 16, bar_h + th + 22),
                          (30, 40, 200), -1)
            cv2.putText(frame, label, (16, bar_h + th + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    # ----------------------------------------------------------- scoring step
    def _on_sample(self, t_sec: float, rgb) -> None:
        emb = self.encoder.encode_images([rgb])[0]
        frame = self.scorer.update(t_sec, emb)
        classes, _ = self.scorer.class_matrix()
        cls_vec = np.array([frame.class_scores.get(c, 0.0) for c in self.bank.classes],
                           dtype=np.float32)

        was_on = self.tracker.on
        closed = self.tracker.update(t_sec, frame.score)

        if self.tracker.on and not was_on:
            # Event just opened: name it, then adopt that class's temporal family
            # so the close criterion matches the kind of event it turned out to be.
            self._active_cls_sum = cls_vec.copy()
            self._active_n = 1
            cls = self.bank.classes[int(np.argmax(cls_vec))] if len(self.bank.classes) else "unknown"
            self.tracker.family = family_for(cls)
            self.active = LiveEvent(cls, self.tracker.start or t_sec, None,
                                    frame.score, last_t=t_sec)
        elif self.tracker.on and self.active is not None:
            self._active_cls_sum += cls_vec
            self._active_n += 1
            self.active.class_name = self.bank.classes[
                int(np.argmax(self._active_cls_sum))] if len(self.bank.classes) else "unknown"
            self.active.peak = max(self.active.peak, frame.score)
            self.active.last_t = t_sec

        if closed is not None and self.active is not None:
            self.active.end = closed[1]
            self.events.append(self.active)
            self.active = None
            self._active_cls_sum, self._active_n = None, 0
            self.tracker.family = FAMILIES["default"]

        # Video-level readout. Intervals come from the onset signal, which is
        # silent on a clip that is anomalous from its first frame - a burning
        # building for six seconds has no onset. Without this the console shows
        # nothing on exactly the clips the detector is most confident about.
        if frame.level1 > self.level1_max:
            self.level1_max = frame.level1
            if len(self.bank.classes):
                self.level1_cls = self.bank.classes[int(np.argmax(cls_vec))]

        self.curve.append((round(t_sec, 2), round(frame.score, 3),
                           round(frame.semantic, 3), round(frame.deviation, 3)))

        arming = None
        if not self.tracker.on:
            cand = getattr(self.tracker, "_cand_start", None)
            if cand is not None:
                arming = {"held": round(max(t_sec - cand, 0.0), 2),
                          "need": round(self.tracker.family.min_on_s, 2)}

        order = np.argsort(-cls_vec)[:5] if len(cls_vec) else []
        with self._lock:
            self._state = {
                "ready": True,
                "t": round(t_sec, 2),
                "score": round(frame.score, 3),
                "semantic": round(frame.semantic, 3),
                "deviation": round(frame.deviation, 3),
                "warm": frame.warm,
                "hi": self.cfg.hi, "lo": self.cfg.lo,
                "top_classes": [{"name": self.bank.classes[i],
                                 "p": round(float(cls_vec[i]), 4)} for i in order],
                "level1": round(frame.level1, 3),
                "level1_max": round(self.level1_max, 3),
                "level1_class": self.level1_cls,
                "level1_flag": bool(self.level1_max >= self.cfg.level1_threshold),
                "level1_threshold": self.cfg.level1_threshold,
                # How close the tracker is to opening an event. The score bar
                # goes hot on a single frame, but evidence must hold above the
                # threshold for min_on_s before anything opens - without this the
                # console looks broken every time a spike fails to persist.
                "arming": arming,
                "active": self.active.as_dict() if self.active else None,
                "events": [e.as_dict() for e in self.events[-25:]],
                "curve": list(self.curve),
                "video": self.video_path.split("/")[-1],
            }

    # ------------------------------------------------------------------- run
    def run(self) -> None:
        import cv2

        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                with self._lock:
                    self._state = {"ready": False, "error": f"cannot open {self.video_path}"}
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if not (0.1 < fps < 1000):
                fps = 25.0
            sample_stride = max(1, int(round(fps / max(self.cfg.sample_hz, 1e-3))))

            self.scorer.reset()
            self.tracker = HysteresisTracker(self.cfg.hi, self.cfg.lo, FAMILIES["default"])
            self.events.clear()
            self.active = None
            self.curve.clear()
            self.level1_max = float("-inf")
            self.level1_cls = None

            idx = 0
            wall0 = time.perf_counter()
            while not self._stop.is_set():
                ok, bgr = cap.read()
                if not ok:
                    break
                t_sec = idx / fps

                if idx % sample_stride == 0:
                    h, w = bgr.shape[:2]
                    s = self.cfg.short_side / max(min(h, w), 1)
                    small = (cv2.resize(bgr, (int(w * s), int(h * s)),
                                        interpolation=cv2.INTER_AREA)
                             if s < 1 else bgr)
                    self._on_sample(t_sec, cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

                st = self.state
                label = None
                if st.get("active"):
                    label = f"{st['active']['class_name']}  {st['active']['duration']:.0f}s"
                shown = self._annotate(bgr, t_sec, st.get("score", 0.0), label)
                ok, buf = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if ok:
                    with self._lock:
                        self._jpeg = buf.tobytes()

                if self.cfg.speed > 0:
                    due = wall0 + t_sec / self.cfg.speed
                    now = time.perf_counter()
                    if due > now:
                        time.sleep(min(due - now, 0.5))
                idx += 1

            cap.release()
            if not self.cfg.loop:
                break
