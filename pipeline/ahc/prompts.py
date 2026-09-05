"""Prompt bank — the label set lives here as *text*, never as weights.

Adding an event type is adding a sentence. Nothing downstream is recompiled,
retrained, or reshaped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Generic templates applied to every class name. Deliberately plain: CLIP-family
# encoders respond better to natural captions than to keyword soup.
#
# Viewpoint words carry real weight and are a known gap: on T020 (debris
# striking a windscreen) the best road-surface phrasing scored +0.084 against a
# normal baseline of +0.160, while "a dashcam view of an object hitting the car"
# scored +0.163 - the only phrasing that cleared it. Adding aerial/dashcam/cctv
# variants to every class was tried and reverted: it doubled the bank, shifted
# every score, and cost D2 9.0 -> 2.3 because the thresholds no longer fitted
# the new scale. The right version is per-class viewpoint phrasings with the
# thresholds refitted, not a blanket template.
TEMPLATES = [
    "a video frame showing {}",
    "surveillance footage of {}",
    "{}",
]

# Optional richer phrasings. A class absent from this map still works — it falls
# back to its own name — so an unseen thirteenth class costs zero code.
DESCRIPTORS: dict[str, list[str]] = {
    "traffic_congestion": [
        "heavy traffic congestion with queued vehicles",
        "a traffic jam, vehicles barely moving",
        "a long queue of stopped cars on a road",
    ],
    "traffic_accident": [
        "a road traffic accident, vehicles collided",
        "a car crash on a street",
        "the aftermath of a vehicle collision",
    ],
    "stalled_or_broken_down_vehicle": [
        "a broken down vehicle stopped on the shoulder of a road",
        "a stalled car halted in a traffic lane",
        "a disabled vehicle with hazard lights on the roadside",
    ],
    "vehicle_blocking_traffic": [
        "a vehicle stopped in the road blocking other traffic",
        "a car obstructing a lane while other vehicles wait",
    ],
    "wrong_way_driving": [
        "a vehicle driving the wrong way against oncoming traffic",
        "a car travelling in the wrong direction on a road",
    ],
    "road_spill_or_debris": [
        "debris or spilled cargo scattered on the road surface",
        "an obstruction of rubble or objects lying on a roadway",
    ],
    "waterlogging_or_flood": [
        "a flooded road submerged in water",
        "waterlogging, vehicles driving through standing water",
    ],
    "fire": [
        "an active fire with visible flames",
        "a building or vehicle on fire",
    ],
    "smoke": [
        "thick smoke rising over an area",
        "a smoke plume from a fire",
    ],
    "fighting_or_violence": [
        "people fighting, a physical altercation",
        "a violent assault between people in public",
    ],
    "loitering_or_suspicious_presence": [
        "a person loitering suspiciously in an area",
        "someone lingering around with suspicious behaviour",
    ],
}

# What "routine" looks like.
#
# This side needs the same structure as the anomaly side, for two reasons.
#
# Arithmetic: the score compares the best-matching anomaly prompt against the
# best-matching normal prompt, and the maximum of many draws is higher than the
# maximum of few regardless of content. Six normal prompts against ninety-eight
# anomaly prompts put a floor under every video. Measured on normal footage,
# that floor alone was enough to raise events.
#
# Coverage: a normal bank that only describes roads leaves every non-road scene
# with nothing to match, so the margin goes high because the scene is
# undescribed rather than because anything is wrong. These are grouped by
# scenario so the comparison is max-over-concepts on both sides.
NORMAL_SCENARIOS: dict[str, list[str]] = {
    "free_flowing_traffic": [
        "a normal street scene with traffic flowing freely",
        "an ordinary road with vehicles driving at normal speed",
        "cars moving steadily along a highway",
    ],
    "empty_road": [
        "a quiet empty road with no vehicles",
        "an ordinary street with light traffic",
    ],
    "parked_vehicles": [
        "cars parked normally in a car park",
        "vehicles parked along the kerb of a street",
    ],
    "pedestrians": [
        "people walking normally along a pavement",
        "pedestrians crossing a road at a crossing",
    ],
    "aerial_city": [
        "a routine aerial view of a city with nothing unusual",
        "an ordinary drone view of buildings and streets",
        "an aerial view of fields and countryside",
    ],
    "surveillance_idle": [
        "an uneventful surveillance camera view",
        "a static security camera view with little happening",
    ],
    "night_scene": [
        "an ordinary street at night with normal traffic",
        "a dark scene at night with nothing unusual happening",
    ],
    "indoor_or_building": [
        "an ordinary indoor scene",
        "the outside of a building with nothing unusual",
    ],
    "open_space": [
        "an ordinary park or open public space",
        "a car park or open yard with nothing unusual",
    ],
    "low_quality_footage": [
        "a blurry low resolution camera view of an ordinary scene",
        "a grainy video of an ordinary place",
    ],
    "junction": [
        "an ordinary road junction with vehicles waiting at a signal",
        "a roundabout with traffic circulating normally",
    ],
}

NORMAL_PROMPTS = [p for group in NORMAL_SCENARIOS.values() for p in group]

NORMAL = "normal"


def humanise(class_name: str) -> str:
    return class_name.replace("_or_", " or ").replace("_", " ").strip()


def phrasings(class_name: str) -> list[str]:
    """Every text string used to represent one class."""
    base = DESCRIPTORS.get(class_name, [humanise(class_name)])
    out: list[str] = []
    for b in base:
        out.append(b)
        for t in TEMPLATES:
            s = t.format(b)
            if s not in out:
                out.append(s)
    return out


@dataclass
class PromptBank:
    """Encoded prompts plus the bookkeeping to group them back by class."""

    texts: list[str]
    labels: list[str]            # class name per text ("normal" for the normal set)
    classes: list[str]           # anomaly class names, ordered
    embeddings: np.ndarray | None = None   # (P, D) L2-normalised

    @property
    def normal_mask(self) -> np.ndarray:
        return np.array([l == NORMAL for l in self.labels], dtype=bool)

    def class_mask(self, cls: str) -> np.ndarray:
        return np.array([l == cls for l in self.labels], dtype=bool)

    def with_embeddings(self, emb: np.ndarray) -> "PromptBank":
        emb = np.asarray(emb, dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return PromptBank(self.texts, self.labels, self.classes, emb)


def build(classes: list[str], extra: dict[str, list[str]] | None = None) -> PromptBank:
    """Assemble a bank for an arbitrary class list, discovered at runtime."""
    texts: list[str] = []
    labels: list[str] = []
    for c in classes:
        ph = list(phrasings(c))
        if extra and c in extra:
            ph += [p for p in extra[c] if p not in ph]
        for p in ph:
            texts.append(p)
            labels.append(c)
    # Expand the normal side with the same templates the classes get, so
    # neither side wins on phrasing count alone.
    for group in NORMAL_SCENARIOS.values():
        for base in group:
            for t in [base] + [tpl.format(base) for tpl in TEMPLATES]:
                if t not in texts:
                    texts.append(t)
                    labels.append(NORMAL)
    return PromptBank(texts=texts, labels=labels, classes=list(classes))


def descriptions_as_prompts(gt, per_class: int = 6,
                            min_len: int = 20) -> dict[str, list[str]]:
    """Mine `description_summary` for prompts.

    This uses the training data the way it should be used: as a source of
    *language*, not as targets for a classifier. The result is still text, so
    the system stays open-set.
    """
    out: dict[str, list[str]] = {}
    if gt is None or len(gt) == 0 or "description_summary" not in gt.columns:
        return out
    sub = gt[gt["class_name"] != NORMAL]
    for cls, grp in sub.groupby("class_name"):
        seen: list[str] = []
        counts = (grp["description_summary"].dropna().astype(str)
                  .str.strip().value_counts())
        for text in counts.index:
            if len(text) < min_len:
                continue
            seen.append(text)
            if len(seen) >= per_class:
                break
        if seen:
            out[str(cls)] = seen
    return out
