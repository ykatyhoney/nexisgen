"""Category validation checks for sampled rows."""

from __future__ import annotations

import base64
import json
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import ClipRecord

logger = logging.getLogger(__name__)

_DEFAULT_STRICT_MODEL = "gemini-3.1-flash-lite-preview"
_CATEGORY_NATURE = "nature"
_STRICT_ALLOWED_WINNERS = {
    "nature",
    "people",
    "animal",
    "vehicle",
    "urban",
    "indoor",
    "other",
}
_STRICT_PROMPT = """
You are validating whether a clip belongs to:
Nature / landscape / scenery.

You are given the middle 3 frames from a 5-second clip.

For each frame, decide which category is dominant:
- nature
- people
- animal
- vehicle
- urban
- indoor
- other

Strict rule:
- PASS for nature only when natural scenery is the main subject.
- If a person is central and dominant, winner must not be nature.
- If an animal is the main subject, winner must not be nature.
- If a vehicle, city/urban scene, or indoor scene dominates, winner must not be nature.

Return JSON only:
{
  "frames": [
    {
      "frame_index": 0,
      "winner": "nature|people|animal|vehicle|urban|indoor|other",
      "nature_score": 0.0,
      "people_score": 0.0,
      "animal_score": 0.0,
      "vehicle_score": 0.0,
      "urban_score": 0.0,
      "indoor_score": 0.0
    },
    {
      "frame_index": 1,
      "winner": "nature|people|animal|vehicle|urban|indoor|other",
      "nature_score": 0.0,
      "people_score": 0.0,
      "animal_score": 0.0,
      "vehicle_score": 0.0,
      "urban_score": 0.0,
      "indoor_score": 0.0
    },
    {
      "frame_index": 2,
      "winner": "nature|people|animal|vehicle|urban|indoor|other",
      "nature_score": 0.0,
      "people_score": 0.0,
      "animal_score": 0.0,
      "vehicle_score": 0.0,
      "urban_score": 0.0,
      "indoor_score": 0.0
    }
  ]
}
"""

_STRONG_NATURE_PHRASES = {
    "natural landscape",
    "scenic landscape",
    "mountain landscape",
    "nature scenery",
    "wide landscape shot",
    "natural scenery",
    "the clip shows a forest",
    "the clip shows a mountain",
    "the scene shows a lake",
    "the main subject is nature",
    "the main subject is a landscape",
    "the main focus is natural scenery",
}
_TRANSIENT_LLM_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_TRANSIENT_LLM_ERROR_HINTS = (
    "timeout",
    "timed out",
    "rate limit",
    "too many requests",
    "temporarily unavailable",
    "internal server error",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "connection error",
    "connection reset",
)


class _FailOpenTransientLLMError(Exception):
    """Transient LLM server-side issue; category check should pass."""


def _is_transient_llm_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and status_code in _TRANSIENT_LLM_STATUS_CODES:
        return True
    lowered = str(error).lower()
    return any(hint in lowered for hint in _TRANSIENT_LLM_ERROR_HINTS)


@dataclass
class FrameResult:
    frame_index: int
    winner: str
    nature_score: float
    people_score: float
    animal_score: float
    vehicle_score: float
    urban_score: float
    indoor_score: float


@dataclass
class StrictPassResult:
    frames: list[FrameResult]


@dataclass
class FinalDecision:
    decision: str
    stage: str
    reason: str
    nature_score: float
    rival_score: float
    margin: float


def clamp_score(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def get_middle_three_frame_paths(frame_paths: list[Path]) -> list[Path] | None:
    existing = [path for path in frame_paths if path.exists()]
    if not existing:
        return None
    unique: list[Path] = []
    seen: set[str] = set()
    for path in existing:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if len(unique) < 3:
        return None
    if len(unique) == 3:
        return unique
    start = max(0, min(len(unique) - 3, (len(unique) // 2) - 1))
    return unique[start : start + 3]


def parse_strict_pass(data: dict[str, Any]) -> StrictPassResult | None:
    frames_raw = data.get("frames", [])
    if not isinstance(frames_raw, list) or len(frames_raw) != 3:
        return None
    frames: list[FrameResult] = []
    for item in frames_raw:
        if not isinstance(item, dict):
            return None
        winner = str(item.get("winner", "")).strip().lower()
        if winner not in _STRICT_ALLOWED_WINNERS:
            return None
        frames.append(
            FrameResult(
                frame_index=int(item.get("frame_index", 0)),
                winner=winner,
                nature_score=clamp_score(item.get("nature_score", 0.0)),
                people_score=clamp_score(item.get("people_score", 0.0)),
                animal_score=clamp_score(item.get("animal_score", 0.0)),
                vehicle_score=clamp_score(item.get("vehicle_score", 0.0)),
                urban_score=clamp_score(item.get("urban_score", 0.0)),
                indoor_score=clamp_score(item.get("indoor_score", 0.0)),
            )
        )
    return StrictPassResult(frames=frames)


def strict_pass_decision(strict_result: StrictPassResult) -> FinalDecision:
    nature_scores = [f.nature_score for f in strict_result.frames]
    people_scores = [f.people_score for f in strict_result.frames]
    animal_scores = [f.animal_score for f in strict_result.frames]
    vehicle_scores = [f.vehicle_score for f in strict_result.frames]
    urban_scores = [f.urban_score for f in strict_result.frames]
    indoor_scores = [f.indoor_score for f in strict_result.frames]

    avg_nature = sum(nature_scores) / len(nature_scores)
    avg_people = sum(people_scores) / len(people_scores)
    avg_animal = sum(animal_scores) / len(animal_scores)
    avg_vehicle = sum(vehicle_scores) / len(vehicle_scores)
    avg_urban = sum(urban_scores) / len(urban_scores)
    avg_indoor = sum(indoor_scores) / len(indoor_scores)

    best_rival = max(avg_people, avg_animal, avg_vehicle, avg_urban, avg_indoor)
    margin = avg_nature - best_rival
    nature_wins = sum(1 for frame in strict_result.frames if frame.winner == _CATEGORY_NATURE)
    reason = (
        f"Nature wins {nature_wins}/3 middle frames, "
        f"avg_nature={avg_nature:.2f}, margin={margin:.2f}"
    )

    if nature_wins >= 2 and avg_nature >= 0.72 and margin >= 0.12:
        return FinalDecision(
            decision="accept",
            stage="strict",
            reason=reason,
            nature_score=avg_nature,
            rival_score=best_rival,
            margin=margin,
        )

    return FinalDecision(
        decision="reject",
        stage="strict",
        reason=reason,
        nature_score=avg_nature,
        rival_score=best_rival,
        margin=margin,
    )


class NatureCategoryChecker:
    """Category checker with caption gate and strict vision pass."""

    def __init__(
        self,
        *,
        enabled: bool,
        api_key: str,
        timeout_sec: int,
        max_samples: int,
        base_url: str | None,
        model: str = _DEFAULT_STRICT_MODEL,
    ):
        self._enabled = enabled
        self._api_key = api_key.strip()
        self._timeout_sec = timeout_sec
        self._max_samples = max(0, max_samples)
        self._base_url = base_url
        self._model = model

    @property
    def active(self) -> bool:
        return self._enabled and self._max_samples > 0

    def check(
        self,
        *,
        sampled: list[ClipRecord],
        frame_paths_by_clip_id: dict[str, list[Path]],
    ) -> list[str]:
        if not self.active:
            return []

        failures: list[str] = []
        checked = 0
        client = None
        for row in sampled:
            if checked >= self._max_samples:
                break

            middle_three = get_middle_three_frame_paths(frame_paths_by_clip_id.get(row.clip_id, []))
            if middle_three is None:
                failures.append(f"category_strict_frames_missing:{row.clip_id}")
                checked += 1
                continue

            if not self._api_key:
                failures.append(f"category_strict_api_key_missing:{row.clip_id}")
                checked += 1
                continue
            if client is None:
                client = self._build_client()
                if client is None:
                    failures.append(f"category_strict_client_unavailable:{row.clip_id}")
                    checked += 1
                    continue
            try:
                parsed = self._run_strict_pass(client=client, frame_paths=middle_three)
            except _FailOpenTransientLLMError as exc:
                logger.warning(
                    "Category strict transient LLM error for clip_id=%s; fail-open pass: %s",
                    row.clip_id,
                    exc,
                )
                checked += 1
                continue
            if parsed is None:
                failures.append(f"category_strict_response_invalid:{row.clip_id}")
                checked += 1
                continue
            decision = strict_pass_decision(parsed)
            if decision.decision == "reject":
                failures.append(f"category_strict_reject:{row.clip_id}")
            checked += 1
        return failures

    def _build_client(self) -> object | None:
        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {
                "api_key": self._api_key,
                "timeout": self._timeout_sec,
            }
            if self._base_url:
                kwargs["base_url"] = self._base_url
            return OpenAI(**kwargs)
        except Exception:
            return None

    def _run_strict_pass(self, *, client: object, frame_paths: list[Path]) -> StrictPassResult | None:
        content: list[dict[str, Any]] = [{"type": "text", "text": _STRICT_PROMPT}]
        for frame_path in frame_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._frame_data_uri(frame_path)},
                }
            )
        try:
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model=self._model,
                messages=[{"role": "user", "content": content}],
                max_tokens=320,
            )
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            text = getattr(message, "content", "") if message is not None else ""
            return self._parse_strict_text(str(text))
        except Exception as exc:
            if _is_transient_llm_error(exc):
                raise _FailOpenTransientLLMError(str(exc)) from exc
            return None

    def _parse_strict_text(self, output_text: str) -> StrictPassResult | None:
        text = output_text.strip()
        if not text:
            return None
        try:
            data = json.loads(text)
            return parse_strict_pass(data)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return parse_strict_pass(data)
        except Exception:
            return None

    def _frame_data_uri(self, frame_path: Path) -> str:
        payload = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"
