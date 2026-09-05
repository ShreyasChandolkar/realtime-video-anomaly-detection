"""Frame sampling and embedding - the one expensive, one-off GPU pass.

Everything downstream reads the cache this writes, never the video. That is what
makes iteration cheap: the embeddings are roughly two orders of magnitude
smaller than the footage, so the whole corpus fits in RAM and a full re-score is
seconds without touching a GPU.

Torch is imported lazily so this module can be imported on a machine that only
needs the path helpers.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

DEFAULT_MODEL = "google/siglip2-base-patch16-224"
FALLBACK_MODELS = ["google/siglip-base-patch16-224", "openai/clip-vit-base-patch16"]


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


# ---------------------------------------------------------------- cache paths
@dataclass
class Cache:
    root: Path
    model_id: str = DEFAULT_MODEL

    @property
    def base(self) -> Path:
        return Path(self.root) / model_slug(self.model_id)

    def emb(self, split: str, video_id: str) -> Path:
        return self.base / "emb" / split / f"{video_id}.npy"

    def meta(self, split: str, video_id: str) -> Path:
        return self.base / "meta" / split / f"{video_id}.json"

    @property
    def prompts(self) -> Path:
        return self.base / "prompts.npz"

    def has(self, split: str, video_id: str) -> bool:
        e, m = self.emb(split, video_id), self.meta(split, video_id)
        return e.exists() and m.exists() and e.stat().st_size > 0

    def load(self, split: str, video_id: str) -> tuple[np.ndarray, dict]:
        emb = np.load(self.emb(split, video_id), mmap_mode="r")
        meta = json.loads(self.meta(split, video_id).read_text())
        return emb, meta


# ------------------------------------------------------------- frame sampling
def sample_frames(path: str | Path, sample_hz: float = 4.0,
                  max_frames: int | None = None,
                  short_side: int = 256) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_sec, RGB frame) at approximately sample_hz.

    Uses grab() to skip cheaply. Decoding every frame only to discard most of
    them is the easiest way to make this stage the bottleneck.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not (0.1 < fps < 1000):
        fps = 25.0
    stride = max(1, int(round(fps / max(sample_hz, 1e-3))))

    idx, taken = 0, 0
    try:
        while True:
            if not cap.grab():
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                h, w = frame.shape[:2]
                if short_side and min(h, w) > short_side:
                    s = short_side / min(h, w)
                    frame = cv2.resize(frame, (int(round(w * s)), int(round(h * s))),
                                       interpolation=cv2.INTER_AREA)
                yield idx / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                taken += 1
                if max_frames and taken >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def probe(path: str | Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"ok": w > 0, "fps": fps, "n_frames": n, "width": w, "height": h,
            "duration_s": (n / fps) if fps > 0 else None}


# ------------------------------------------------------------------- encoder
class FrameEncoder:
    """Frozen image/text tower. Inference only - nothing here is ever trained."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None,
                 batch_size: int = 64, dtype: str = "auto"):
        import torch
        from transformers import AutoModel, AutoProcessor

        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.batch_size = batch_size
        if dtype == "auto":
            td = torch.float16 if device == "cuda" else torch.float32
        else:
            td = getattr(torch, dtype)
        self.torch_dtype = td

        last_err = None
        for mid in [model_id] + [m for m in FALLBACK_MODELS if m != model_id]:
            try:
                self.processor = AutoProcessor.from_pretrained(mid)
                self.model = AutoModel.from_pretrained(mid, torch_dtype=td).to(device).eval()
                self.model_id = mid
                break
            except Exception as e:
                last_err = e
        else:
            raise RuntimeError(f"no usable encoder: {last_err}")

        cfg = self.model.config
        self.dim = int(getattr(cfg, "projection_dim", 0)
                       or getattr(getattr(cfg, "text_config", cfg), "hidden_size", 768))

    @property
    def is_siglip(self) -> bool:
        return "siglip" in self.model_id.lower()

    @staticmethod
    def _as_tensor(x):
        """transformers 5.x returns an output object here; 4.x returned a tensor."""
        if hasattr(x, "float"):
            return x
        for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
            v = getattr(x, attr, None)
            if v is not None:
                return v.mean(dim=1) if attr == "last_hidden_state" and v.ndim == 3 else v
        raise TypeError(f"cannot extract embedding tensor from {type(x)}")

    def _norm(self, x) -> np.ndarray:
        v = self._as_tensor(x).float().cpu().numpy()
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        return v.astype(np.float32)

    def encode_images(self, frames: list) -> np.ndarray:
        import torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(frames), self.batch_size):
                chunk = frames[i:i + self.batch_size]
                inputs = self.processor(images=chunk, return_tensors="pt")
                moved = {}
                for k, v in inputs.items():
                    moved[k] = (v.to(self.device, self.torch_dtype)
                                if v.is_floating_point() else v.to(self.device))
                out.append(self._norm(self.model.get_image_features(**moved)))
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch
        pad = "max_length" if self.is_siglip else True  # siglip needs max_length
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), 64):
                inputs = self.processor(text=texts[i:i + 64], padding=pad,
                                        truncation=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                out.append(self._norm(self.model.get_text_features(**inputs)))
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)


# ----------------------------------------------------------------- extraction
def extract_video(encoder: FrameEncoder, cache: Cache, split: str, video_id: str,
                  path: str | Path, sample_hz: float = 4.0,
                  max_frames: int | None = None, overwrite: bool = False) -> dict:
    if not overwrite and cache.has(split, video_id):
        meta = json.loads(cache.meta(split, video_id).read_text())
        meta["skipped"] = True
        return meta

    t0 = time.perf_counter()
    info = probe(path)
    times, frames, err = [], [], None
    try:
        for t, fr in sample_frames(path, sample_hz, max_frames):
            times.append(t)
            frames.append(fr)
    except Exception as e:
        err = str(e)

    emb = encoder.encode_images(frames) if frames else np.zeros((0, encoder.dim), np.float32)
    dt = time.perf_counter() - t0

    cache.emb(split, video_id).parent.mkdir(parents=True, exist_ok=True)
    cache.meta(split, video_id).parent.mkdir(parents=True, exist_ok=True)
    np.save(cache.emb(split, video_id), emb.astype(np.float16))
    meta = {
        "video_id": video_id, "split": split, "path": str(path),
        "model_id": encoder.model_id,
        "dim": int(emb.shape[1]) if emb.size else encoder.dim,
        "sample_hz": sample_hz,
        "effective_hz": (round(len(times) / (times[-1] - times[0]), 3)
                         if len(times) > 1 and times[-1] > times[0] else None),
        "n_samples": int(len(times)),
        "timestamps": [round(float(t), 4) for t in times],
        "video": info, "error": err, "extract_s": round(dt, 3),
        "fps_processed": round(len(times) / dt, 1) if dt > 0 else None,
    }
    cache.meta(split, video_id).write_text(json.dumps(meta))
    return meta


def load_prompt_bank(cache: Cache):
    """Read the cached prompt embeddings. No torch, so this works on any machine."""
    from .prompts import PromptBank

    p = cache.prompts
    if not p.exists():
        raise FileNotFoundError(f"no prompt cache at {p} - run extract_features first")
    z = np.load(p, allow_pickle=True)
    texts = [str(t) for t in z["texts"]]
    labels = [str(l) for l in z["labels"]]
    classes = sorted({l for l in labels if l != "normal"})
    return PromptBank(texts, labels, classes).with_embeddings(z["embeddings"])


def encode_prompt_bank(encoder: FrameEncoder, bank, cache: Cache):
    """Encode once, cache keyed by text content so staleness is impossible."""
    h = hashlib.sha1(" ".join(bank.texts).encode()).hexdigest()[:12]
    p = cache.prompts
    if p.exists():
        try:
            z = np.load(p, allow_pickle=True)
            if str(z["hash"]) == h and str(z["model"]) == encoder.model_id:
                return bank.with_embeddings(z["embeddings"])
        except Exception:
            pass
    emb = encoder.encode_texts(bank.texts)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, embeddings=emb, hash=h, model=encoder.model_id,
             texts=np.array(bank.texts, dtype=object),
             labels=np.array(bank.labels, dtype=object))
    return bank.with_embeddings(emb)
