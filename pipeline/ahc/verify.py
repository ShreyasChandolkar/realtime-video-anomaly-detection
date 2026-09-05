"""Tier 2: VLM verification and naming, on candidates only.

Retrieval already finds the right moments - class-agnostic event F1 is roughly
three times the class-aware figure - so the failure is naming, not
localisation. That shapes the job given to the VLM:

  not  "what is happening in this video"        (open classification, hard)
  but  "which of these three is it, or none"    (forced choice, easy)

The shortlist comes from the cheap stage, whose top-3 contains the right answer
about two thirds of the time. Framing it this way is what lets a 3B model do the
work of a much larger one, and it is why the stage stays affordable: it runs
once per detected event, not once per frame.

The VLM may also answer "none", which is how a false positive gets retracted
and how an event outside the prompt bank stays describable in free text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


@dataclass
class Verdict:
    class_name: str | None      # None means the VLM rejected every candidate
    confidence: float
    description: str
    raw: str = ""
    latency_ms: float = 0.0

    @property
    def rejected(self) -> bool:
        return self.class_name is None

    def as_dict(self) -> dict:
        return {"class_name": self.class_name, "confidence": round(self.confidence, 3),
                "description": self.description, "rejected": self.rejected,
                "latency_ms": round(self.latency_ms, 1)}


def humanise(c: str) -> str:
    return c.replace("_or_", " or ").replace("_", " ")


PROMPT = """You are reviewing a short clip from a surveillance or drone camera.

A detector flagged something unusual here. Your job is to decide which of these \
candidates it actually is, or to reject all of them:

{options}
{none_option}. none of these - the clip is ordinary

Judge only what is visible. A stationary vehicle in a parking area is ordinary; \
one stopped in a traffic lane is not. Ordinary traffic, however busy, is not \
congestion unless vehicles are queued or barely moving.

Reply with JSON only:
{{"choice": <number>, "confidence": <0.0-1.0>, "description": "<one short sentence>"}}"""


class VlmVerifier:
    """Loads lazily so importing this module costs nothing without a GPU."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cuda",
                 max_frames: int = 8, max_pixels: int = 401408):
        self.model_id = model_id
        self.device = device
        self.max_frames = max_frames
        self.max_pixels = max_pixels
        self._model = None
        self._proc = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as Model
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as Model

        self._proc = AutoProcessor.from_pretrained(self.model_id,
                                                   max_pixels=self.max_pixels)
        self._model = Model.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        ).eval()

    @staticmethod
    def _parse(text: str, candidates: list[str]) -> tuple[str | None, float, str]:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                choice = int(d.get("choice", 0))
                conf = float(d.get("confidence", 0.5))
                desc = str(d.get("description", "")).strip()
                if 1 <= choice <= len(candidates):
                    return candidates[choice - 1], conf, desc
                return None, conf, desc
            except Exception:
                pass
        # Fall back to a name match if the model ignored the format.
        low = text.lower()
        for c in candidates:
            if humanise(c).lower() in low or c.lower() in low:
                return c, 0.4, text.strip()[:160]
        return None, 0.3, text.strip()[:160]

    def verify(self, frames: list, candidates: list[str],
               shuffle_seed: int | None = None) -> Verdict:
        """frames: RGB numpy arrays sampled from the event window.

        Candidates are presented in a shuffled order. Given them ranked, the
        model chose option 1 every time at 0.95 confidence and agreed with the
        detector on every case - including the four it got wrong. Position bias,
        not judgement.

        The seed varies per call rather than being fixed: a constant seed
        permutes the ranks the same way every time, so position 1 always holds
        the same rank and one systematic bias simply replaces another.
        """
        import random
        import time

        import torch
        from PIL import Image

        self.load()
        t0 = time.perf_counter()

        order = list(range(len(candidates)))
        seed = (shuffle_seed if shuffle_seed is not None
                else abs(hash(tuple(candidates))) % (2 ** 31))
        random.Random(seed).shuffle(order)
        shown = [candidates[i] for i in order]

        opts = "\n".join(f"{i+1}. {humanise(c)}" for i, c in enumerate(shown))
        prompt = PROMPT.format(options=opts, none_option=len(candidates) + 1)

        pil = [Image.fromarray(f) for f in frames[: self.max_frames]]
        messages = [{"role": "user", "content":
                     [{"type": "image", "image": im} for im in pil]
                     + [{"type": "text", "text": prompt}]}]
        text = self._proc.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        inputs = self._proc(text=[text], images=pil, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=128, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        raw = self._proc.decode(gen, skip_special_tokens=True)

        cls, conf, desc = self._parse(raw, shown)
        return Verdict(cls, conf, desc, raw.strip(),
                       (time.perf_counter() - t0) * 1000)


def shortlist(class_scores: np.ndarray, classes: list[str], k: int = 5) -> list[str]:
    """Top-k classes over an event window. The cheap stage's top-3 holds the
    right answer roughly two thirds of the time, which is what makes the forced
    choice tractable."""
    if not len(classes):
        return []
    mean = class_scores.mean(axis=0) if class_scores.ndim == 2 else class_scores
    return [classes[i] for i in np.argsort(-mean)[:k]]


def sample_event_frames(video_path: str, start: float, end: float,
                        n: int = 8, short_side: int = 336) -> list:
    """Evenly spaced frames across the event window."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not (0.1 < fps < 1000):
        fps = 25.0
    times = np.linspace(start, max(end, start + 0.1), n)
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, bgr = cap.read()
        if not ok:
            continue
        h, w = bgr.shape[:2]
        s = short_side / max(min(h, w), 1)
        if s < 1:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out
