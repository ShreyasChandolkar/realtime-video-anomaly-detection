"""Held-out discipline, enforced in code rather than remembered.

The public test set is the only honest estimate of how this behaves on the
private set, and it is worth exactly one look. Every time a threshold is chosen
against it, that estimate degrades. So reading test requires an explicit unlock,
and calibration reads train only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "configs/calibration.json"
TEST_SPLITS = {"test"}


class HeldOutError(RuntimeError):
    """Raised when test data is touched without an explicit unlock."""


def guard_splits(splits: list[str], allow_test: bool) -> list[str]:
    """Refuse to evaluate on test unless deliberately unlocked."""
    touching = [s for s in splits if s in TEST_SPLITS]
    if touching and not allow_test:
        raise HeldOutError(
            f"refusing to read {touching}: the public test set is held out.\n"
            "Calibrate on train. Pass --allow-test only for a final, reported\n"
            "measurement - every look at it costs you the estimate it provides."
        )
    return splits


@dataclass
class Calibration:
    """Everything fitted from data, kept out of the source."""

    hi: float = 0.4
    lo: float = 0.16
    level1_threshold: float = 0.40
    fitted_on: str = "none"
    n_videos: int = 0
    n_events: int = 0
    notes: str = "defaults - not yet fitted"

    @classmethod
    def load(cls, path: Path | None = None) -> "Calibration":
        p = Path(path or CONFIG)
        if not p.exists():
            return cls()
        return cls(**json.loads(p.read_text()))

    def save(self, path: Path | None = None) -> Path:
        p = Path(path or CONFIG)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return p

    @property
    def is_fitted(self) -> bool:
        return self.fitted_on not in ("none", "")

    def describe(self) -> str:
        if not self.is_fitted:
            return "calibration: DEFAULTS (not fitted - numbers are guesses)"
        return (f"calibration: fitted on {self.fitted_on} "
                f"({self.n_videos} videos, {self.n_events} events) | "
                f"hi={self.hi:.2f} lo={self.lo:.2f} L1={self.level1_threshold:.2f}")
