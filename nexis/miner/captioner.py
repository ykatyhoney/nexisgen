"""Caption generation wrapper for miner pipeline."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)


class Captioner:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_sec: int = 30,
        *,
        provider: str = "openai",
        base_url: str | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._timeout_sec = timeout_sec
        self._provider = provider
        self._base_url = base_url

    def _fallback_caption(self) -> str:
        return "A short five second video clip with visible motion."

    def _frame_data_uri(self, frame_path: Path) -> str:
        raw = frame_path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def caption_clip(
        self,
        clip_path: Path,
        source_url: str,
        first_frame_path: Path | None = None,
        frame_paths: list[Path] | None = None,
    ) -> str:
        # Fallback path when API key is not configured.
        if not self._api_key:
            logger.warning(
                "caption fallback used reason=missing_%s_api_key",
                self._provider,
            )
            return self._fallback_caption()

        try:
            client_kwargs: dict[str, object] = {
                "api_key": self._api_key,
                "timeout": self._timeout_sec,
            }
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            client = OpenAI(**client_kwargs)
            valid_frames = [p for p in (frame_paths or []) if p.exists()]
            if not valid_frames and first_frame_path and first_frame_path.exists():
                valid_frames = [first_frame_path]
            # Keep request size controlled while still using >10 timeline frames.
            valid_frames = valid_frames[:12]
            logger.debug("caption request clip=%s frames=%d", clip_path.name, len(valid_frames))
            prompt = (
                "Write one concise training caption for this 5-second video clip. "
                "You will receive timeline-sampled frames from the same clip. "
                "Use all provided frames to describe visible scene and motion in one sentence. "
                "Describe only concrete visual content and do not speculate. "
                f"Clip file name: {clip_path.name}. Source URL: {source_url}."
            )
            frames: list[dict[str, object]] = []
            for frame_path in valid_frames:
                frames.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._frame_data_uri(frame_path)},
                    }
                )
            content: list[dict[str, object]] = [{"type": "text", "text": prompt}] + frames
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": content}],
                max_tokens=150,
            )
            text = (response.choices[0].message.content or "").strip()
            logger.debug("caption response clip=%s text_len=%d", clip_path.name, len(text))
            return text if text else self._fallback_caption()
        except Exception:
            logger.exception("caption generation failed clip=%s", clip_path.name)
            return self._fallback_caption()

