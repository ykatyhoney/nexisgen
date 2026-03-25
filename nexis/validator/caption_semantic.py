"""Semantic caption-to-frame validation for sampled rows."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from ..models import ClipRecord
import logging
logger = logging.getLogger(__name__)

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
            frame_paths = [
                path
                for path in frame_paths_by_clip_id.get(row.clip_id, [])
                if path.exists()
            ]
            if not frame_paths:
                continue
            verdict = self._judge_match(
                client=client,
                caption=row.caption,
                frame_paths=frame_paths[:12],
            )
            if verdict is False:
                failures.append(f"caption_semantic_mismatch:{row.clip_id}")
            checked += 1
        return failures

    def _judge_match(self, *, client: object, caption: str, frame_paths: list[Path]) -> bool | None:
        prompt = (
            "You are validating whether a caption is semantically consistent with timeline-sampled "
            "frames from a short video clip. "
            "Return JSON only: {\"match\": true} or {\"match\": false}. "
            "Use false only for clear mismatch."
            f"\nCaption: {caption}"
        )
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
        except Exception:
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

