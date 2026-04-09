from __future__ import annotations

from nexis.cli import _parse_exclude_hotkeys


def test_parse_exclude_hotkeys() -> None:
    assert _parse_exclude_hotkeys("") == set()
    assert _parse_exclude_hotkeys("  ") == set()
    assert _parse_exclude_hotkeys("a,b, c ,,") == {"a", "b", "c"}

