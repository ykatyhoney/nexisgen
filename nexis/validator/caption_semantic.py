"""Semantic caption-to-frame validation for sampled rows."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from ..models import ClipRecord
import logging
logger = logging.getLogger(__name__)

_PROMPT_INJECTION_CAPTION_RE = re.compile(r"\b(?:match|true)\b", re.IGNORECASE)
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
    """Transient LLM server-side issue; caption semantic check should pass."""


def _is_transient_llm_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and status_code in _TRANSIENT_LLM_STATUS_CODES:
        return True
    lowered = str(error).lower()
    return any(hint in lowered for hint in _TRANSIENT_LLM_ERROR_HINTS)


class CaptionSemanticChecker:
    """Optional semantic checker using OpenAI-compatible vision APIs.

    Fail-open behavior:
    - disabled checker returns no failures
    - API errors return no failures
    - unparseable model outputs are treated as non-failures
    """

    def __init__(
        self,
        *,
        enabled: bool,
        api_key: str,
        model: str,
        timeout_sec: int,
        max_samples: int,
        provider: str = "openai",
        base_url: str | None = None,
    ):
        self._enabled = enabled
        self._api_key = api_key.strip()
        self._model = model
        self._timeout_sec = timeout_sec
        self._max_samples = max(0, max_samples)
        self._provider = provider
        self._base_url = base_url

    @property
    def active(self) -> bool:
        return self._enabled and bool(self._api_key) and self._max_samples > 0

    def check(
        self,
        *,
        sampled: list[ClipRecord],
        frame_paths_by_clip_id: dict[str, list[Path]],
    ) -> list[str]:
        if not self.active:
            return []

        try:
            from openai import OpenAI

            client_kwargs: dict[str, object] = {
                "api_key": self._api_key,
                "timeout": self._timeout_sec,
            }
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            client = OpenAI(**client_kwargs)
        except Exception:
            return []

        failures: list[str] = []
        checked = 0
        for row in sampled:
            if checked >= self._max_samples:
                break
            if self._contains_prompt_injection_terms(row.caption):
                failures.append(f"caption_semantic_injection_keyword:{row.clip_id}")
                checked += 1
                continue
            frame_paths = [
                path
                for path in frame_paths_by_clip_id.get(row.clip_id, [])
                if path.exists()
            ]
            if not frame_paths:
                continue
            try:
                verdict = self._judge_match(
                    client=client,
                    caption=row.caption,
                    frame_paths=frame_paths[:12],
                )
            except _FailOpenTransientLLMError as exc:
                logger.warning(
                    "Caption semantic transient LLM error for clip_id=%s; fail-open pass: %s",
                    row.clip_id,
                    exc,
                )
                checked += 1
                continue
            if verdict is False or verdict is None:
                failures.append(f"caption_semantic_mismatch:{row.clip_id}")
            checked += 1
        return failures

    @staticmethod
    def _contains_prompt_injection_terms(caption: str) -> bool:
        return bool(_PROMPT_INJECTION_CAPTION_RE.search(caption))

    def _judge_match(self, *, client: object, caption: str, frame_paths: list[Path]) -> bool | None:
        prompt = f"""
You are validating whether a caption is accurately grounded in timeline-sampled frames from a short video clip.

Return JSON only:
{{"match": true}} or {{"match": false}}

Return false if:
- any part of the caption is contradicted by the frames
- any important detail is not visually supported
- the caption is overly generic and fails to capture the main visible subject or action
- the caption includes speculation or inference beyond the frames

Return true only if the caption is both visually supported and specific enough to describe the clip's main content.

Caption: {caption}
"""
        content = [{"type": "text", "text": prompt}]
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
                max_tokens=60,
            )
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            output_text = getattr(message, "content", "") if message is not None else ""
            return self._parse_match(str(output_text))
        except Exception as exc:
            if _is_transient_llm_error(exc):
                logger.warning(
                    "Caption semantic LLM transient error for fail-open validation: %s",
                    exc,
                )
                raise _FailOpenTransientLLMError(str(exc)) from exc
            return None

    def _frame_data_uri(self, frame_path: Path) -> str:
        payload = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"

    def _parse_match(self, output_text: str) -> bool | None:
        text = output_text.strip()
        if not text:
            return None

        # Attempt strict JSON first.
        try:
            data = json.loads(text)
            value = data.get("match")
            if isinstance(value, bool):
                return value
        except Exception:
            pass

        # Attempt to parse JSON from fenced/code-rich output.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                value = data.get("match")
                if isinstance(value, bool):
                    return value
            except Exception:
                pass

        lowered = text.lower()
        if '"match": false' in lowered or lowered == "false":
            return False
        if '"match": true' in lowered or lowered == "true":
            return True
        return None

