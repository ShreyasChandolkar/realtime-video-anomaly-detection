"""Dataset discovery and ground-truth loading.

Nothing here assumes a fixed class list beyond what the CSVs contain, and no
path is hardcoded: the dataset root is always passed in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

GT_COLUMNS = [
    "video_id", "level", "is_anomaly", "class_name",
    "start_time_sec", "end_time_sec", "description_summary",
]


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _coerce_gt(df: pd.DataFrame) -> pd.DataFrame:
    for col in GT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["video_id"] = df["video_id"].astype(str)
    df["class_name"] = df["class_name"].fillna("normal").astype(str).str.strip()
    for col in ("start_time_sec", "end_time_sec"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["is_anomaly"].isna().all():
        df["is_anomaly"] = (df["class_name"] != "normal").astype(int)
    else:
        df["is_anomaly"] = (
            pd.to_numeric(df["is_anomaly"], errors="coerce").fillna(0).astype(int)
        )
    df["level"] = pd.to_numeric(df["level"], errors="coerce").fillna(1).astype(int)
    return df[GT_COLUMNS]


@dataclass
class Split:
    """One split (a train class folder, or the test folder)."""

    name: str
    root: Path
    videos: pd.DataFrame
    ground_truth: pd.DataFrame
    source_class: str | None = None


def _resolve_video_paths(videos: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Map whatever the videos.csv filename column is onto real paths."""
    cand_cols = [c for c in videos.columns if c.lower() in
                 ("file_name", "filename", "file", "path", "video_path", "video_file")]
    col = cand_cols[0] if cand_cols else videos.columns[-1]
    vid_dir = root / "videos"
    paths = []
    for raw in videos[col].astype(str):
        name = Path(raw).name
        p = vid_dir / name
        if not p.exists():
            hits = list(vid_dir.glob(f"{Path(name).stem}.*")) if vid_dir.is_dir() else []
            p = hits[0] if hits else vid_dir / name
        paths.append(str(p))
    videos = videos.copy()
    videos["video_path"] = paths
    videos["exists"] = [Path(p).exists() for p in paths]
    id_cols = [c for c in videos.columns if c.lower() in ("video_id", "id")]
    videos["video_id"] = videos[id_cols[0]].astype(str) if id_cols else [
        Path(p).stem for p in paths
    ]
    return videos


VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def _scan_videos(root: Path) -> pd.DataFrame:
    """Enumerate videos/ directly. The manifest is a convenience, not a
    dependency: a folder of footage with no CSV at all must still be
    processable, which is also what running on unseen video requires."""
    vid_dir = root / "videos"
    files = sorted(p for p in vid_dir.iterdir()
                   if p.suffix.lower() in VIDEO_SUFFIXES) if vid_dir.is_dir() else []
    return pd.DataFrame({
        "video_id": [p.stem for p in files],
        "video_path": [str(p) for p in files],
        "exists": [True] * len(files),
    })


def load_split(root: Path, name: str, source_class: str | None = None) -> Split | None:
    root = Path(root)
    v, g = root / "videos.csv", root / "ground_truth.csv"

    videos = pd.DataFrame()
    if v.exists():
        try:
            videos = _resolve_video_paths(_read_csv(v), root)
        except Exception:
            videos = pd.DataFrame()
    # Fall back to the filesystem when the manifest is missing or unusable, and
    # merge in anything on disk the manifest failed to mention.
    scanned = _scan_videos(root)
    if videos.empty or not videos.get("exists", pd.Series(dtype=bool)).any():
        videos = scanned
    elif not scanned.empty:
        missing = scanned[~scanned["video_id"].isin(videos["video_id"])]
        if len(missing):
            videos = pd.concat([videos, missing], ignore_index=True)
    if videos.empty:
        return None

    gt = _coerce_gt(_read_csv(g)) if g.exists() else _coerce_gt(pd.DataFrame())
    return Split(name=name, root=root, videos=videos,
                 ground_truth=gt, source_class=source_class)


def discover(dataset_root: str | Path) -> dict[str, Split]:
    """Walk the delivered layout: train/<class>/ and test/."""
    dataset_root = Path(dataset_root).expanduser()
    splits: dict[str, Split] = {}

    test = load_split(dataset_root / "test", "test")
    if test is not None:
        splits["test"] = test

    train_root = dataset_root / "train"
    if train_root.is_dir():
        for cls_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
            s = load_split(cls_dir, f"train/{cls_dir.name}", source_class=cls_dir.name)
            if s is not None:
                splits[s.name] = s
    return splits


def build_index(dataset_root: str | Path) -> pd.DataFrame:
    """One row per video across every split — the master table."""
    rows = []
    for name, split in discover(dataset_root).items():
        gt = split.ground_truth
        by_video = {vid: sub for vid, sub in gt.groupby("video_id")} if len(gt) else {}
        for _, r in split.videos.iterrows():
            sub = by_video.get(r["video_id"])
            rows.append({
                "video_id": r["video_id"],
                "split": name,
                "group": "test" if name == "test" else "train",
                "source_class": split.source_class,
                "video_path": r["video_path"],
                "exists": bool(r["exists"]),
                "n_events": 0 if sub is None else int((sub["class_name"] != "normal").sum()),
                "classes": "" if sub is None else "|".join(
                    sorted(set(sub.loc[sub["class_name"] != "normal", "class_name"]))
                ),
                "is_anomaly": 0 if sub is None else int(sub["is_anomaly"].max()),
            })
    return pd.DataFrame(rows)


def all_ground_truth(dataset_root: str | Path) -> pd.DataFrame:
    frames = []
    for name, split in discover(dataset_root).items():
        gt = split.ground_truth.copy()
        gt["split"] = name
        gt["group"] = "test" if name == "test" else "train"
        frames.append(gt)
    if not frames:
        return _coerce_gt(pd.DataFrame())
    return pd.concat(frames, ignore_index=True)


def class_vocabulary(gt: pd.DataFrame) -> list[str]:
    """Label set observed in the data — never hardcoded."""
    return sorted(set(gt["class_name"].dropna()) - {"normal"})
