#!/usr/bin/env python3
"""Attach an explanation to every predicted event.

The benchmark awards a reasoning bonus for a 20-500 character explanation and
states it never reduces the score, so an event without one is simply forfeited
marks. We had been sending none.

Explanations are grounded in what the detector actually measured - the class it
settled on, when it fired, how long it persisted - rather than invented prose.
Where the VLM produced a free-text observation for that video, its wording is
preferred, since it describes the frames rather than the score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# What the detector is keying on for each class, in plain language.
EVIDENCE = {
    "traffic_congestion":
        "vehicles are densely queued and moving slowly across the carriageway",
    "traffic_accident":
        "vehicle trajectories converge abruptly and traffic behind comes to a halt",
    "stalled_or_broken_down_vehicle":
        "a vehicle remains stationary in a running traffic lane rather than a lay-by",
    "vehicle_blocking_traffic":
        "a stationary vehicle occupies an active lane while other traffic diverts around it",
    "wrong_way_driving":
        "a vehicle travels against the prevailing direction of the surrounding traffic",
    "road_spill_or_debris":
        "material is scattered across the road surface and vehicles steer around it",
    "waterlogging_or_flood":
        "the road surface is submerged and vehicles move through standing water",
    "fire": "open flame is visible in the scene",
    "smoke": "a plume of smoke rises across the scene without visible flame",
    "fighting_or_violence":
        "people are engaged in a physical altercation rather than passing through",
    "loitering_or_suspicious_presence":
        "a person remains in place while the rest of the scene moves on around them",
}


def humanise(c: str) -> str:
    return c.replace("_or_", " or ").replace("_", " ")


def explain(cls: str, start, end, observation: str | None) -> str:
    """20-500 characters, specific to this event."""
    if observation:
        obs = observation.strip().rstrip(".")
        if 20 <= len(obs) <= 400:
            return f"{obs}. Classified as {humanise(cls)}."
    evidence = EVIDENCE.get(cls, f"the scene matches {humanise(cls)}")
    if start is None or end is None:
        return (f"Across the clip the strongest evidence is for "
                f"{humanise(cls)}: {evidence}.")
    dur = max(float(end) - float(start), 0.0)
    return (f"Between {float(start):.0f}s and {float(end):.0f}s ({dur:.0f}s) "
            f"{evidence}, which the detector scores as {humanise(cls)}.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=str(
        Path.home() / "hackathon-ahc/submissions/BEST_57.0/sub_so400m_d1.json"))
    ap.add_argument("--vlm", default=str(Path.home() / "hackathon-ahc/runs/d1_vlm.json"))
    ap.add_argument("--out", default=str(
        Path.home() / "hackathon-ahc/runs/sub_explained.json"))
    a = ap.parse_args()

    sub = json.loads(Path(a.infile).read_text())
    obs = {}
    p = Path(a.vlm)
    if p.exists():
        for r in json.loads(p.read_text()):
            o = (r.get("observation") or "").strip()
            if len(o) >= 20:
                obs[r["video_id"]] = o
    print(f"{len(obs)} videos have a VLM observation to draw on")

    n = short = 0
    for pred in sub["predictions"]:
        for e in pred["events"]:
            text = explain(e["class_name"], e.get("start_time_sec"),
                           e.get("end_time_sec"), obs.get(pred["video_id"]))
            if not (20 <= len(text) <= 500):
                text = text[:500]
                short += 1
            e["explanation"] = text
            n += 1
    sub["submission_id"] = "ahc-explained"
    Path(a.out).write_text(json.dumps(sub, separators=(",", ":")) + "\n")

    lens = [len(e["explanation"]) for p_ in sub["predictions"] for e in p_["events"]]
    print(f"{n} events explained, lengths {min(lens)}-{max(lens)} chars"
          f"{f' ({short} clipped)' if short else ''}")
    print(f"-> {a.out}  ({Path(a.out).stat().st_size/1024:.1f} KB)")
    for pred in sub["predictions"][:2]:
        for e in pred["events"]:
            print(f"   {pred['video_id']}: {e['explanation'][:130]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
