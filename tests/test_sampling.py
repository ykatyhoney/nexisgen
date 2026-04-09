from __future__ import annotations

from nexis.validator.sampling import select_miners, select_row_indices


def test_deterministic_miner_sampling() -> None:
    miners = [f"hk{i}" for i in range(20)]
    seed = "seed1"
    a = select_miners(miners, seed)
    b = select_miners(miners, seed)
    assert a == b


def test_deterministic_row_sampling() -> None:
    rows = 100
    hk = "hk1"
    seed = "seed2"
    a = select_row_indices(rows, hk, seed)
    b = select_row_indices(rows, hk, seed)
    assert a == b
    assert len(a) <= 3


def test_row_sampling_uses_validator_hotkey() -> None:
    rows = 100
    miner_hotkey = "miner1"
    seed = "seed2"
    validator_a = "validatorA"
    validator_b = "validatorB"

    a1 = select_row_indices(rows, miner_hotkey, seed, validator_a)
    a2 = select_row_indices(rows, miner_hotkey, seed, validator_a)
    b = select_row_indices(rows, miner_hotkey, seed, validator_b)

    assert a1 == a2
    assert a1 != b

