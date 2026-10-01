"""synthetic stub: loudness mastering helper (placeholder, not the original).

``pre_bake_loudnorm`` returns the input path unchanged; ``ALIMITER_SUFFIX`` is empty.
Replace with a real two-pass loudnorm (target -16 LUFS) implementation.
"""
from __future__ import annotations
from pathlib import Path

ALIMITER_SUFFIX = ""  # synthetic stub: no limiter filter appended


def pre_bake_loudnorm(src, target_lufs: float = -16.0):
    """Placeholder: returns ``src`` unchanged."""
    return Path(src)
