"""Deterministic miner and row sampling for validators."""

from __future__ import annotations

import hashlib
import math
import random

from ..protocol import (
    MINER_SAMPLE_MAX,
    MINER_SAMPLE_MIN,
    MINER_SAMPLE_RATE,
    ROW_SAMPLE_ALL_THRESHOLD,
    ROW_SAMPLE_MAX,
    ROW_SAMPLE_RATE,
)


def select_miners(active_hotkeys: list[str], interval_seed: str) -> list[str]:
    if not active_hotkeys:
        return []
    rate_k = int(math.ceil(len(active_hotkeys) * MINER_SAMPLE_RATE))
    k = min(max(MINER_SAMPLE_MIN, rate_k), MINER_SAMPLE_MAX, len(active_hotkeys))
    ranked = sorted(active_hotkeys, key=lambda hk: _det_tiebreak(interval_seed, hk))
    return ranked[:k]


def select_row_indices(
    total_rows: int,
    hotkey: str,
    interval_seed: str,
    validator_hotkey: str | None = None,
) -> list[int]:
    if total_rows <= ROW_SAMPLE_ALL_THRESHOLD:
        return list(range(total_rows))
    k = max(1, int(math.ceil(total_rows * ROW_SAMPLE_RATE)))
    k = min(k, ROW_SAMPLE_MAX, total_rows)
    if validator_hotkey:
        seed_text = f"{hotkey}:{validator_hotkey}:{interval_seed}:{total_rows}"
    else:
        seed_text = f"{hotkey}:{interval_seed}:{total_rows}"
    rng = random.Random(_seed_int(seed_text))
    population = list(range(total_rows))
    rng.shuffle(population)
    return sorted(population[:k])


def _seed_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _det_tiebreak(seed: str, hotkey: str) -> int:
    return _seed_int(f"{seed}:{hotkey}")

