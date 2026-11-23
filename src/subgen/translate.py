"""Context-aware Cloud LLM subtitle translation (Japanese / English -> Simplified Chinese)."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Protocol

from subgen.config import DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL, load_settings
from subgen.models import SubtitleSegment, TranslatorName

ProgressCallback = Callable[[float, str], None]

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class TranslationError(RuntimeError):
    """Raised when subtitle translation fails."""


class LLMClient(Protocol):
    """Minimal protocol for a JSON-returning LLM translation backend."""

    name: str
    model: str

    def complete_json(self, system_prompt: str, user_prompt: str) -> str: ...


class GeminiLLMClient:
    """Google Gemini translation client using `google-genai`."""

    name = "gemini"

    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL) -> None:
        if not api_key:
            raise TranslationError(
                "Gemini API key is missing. Set GEMINI_API_KEY in .env or pass --api-key."
            )
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self.model = model

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "temperature": 0.2,
        }
        # Keep thinking latency & output token cost low on Gemini 4 models for subtitle batches.
        if self.model.lower().startswith("gemini-4") and hasattr(self._types, "ThinkingConfig"):
            config_kwargs["thinking_config"] = self._types.ThinkingConfig(thinking_level="LOW")

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=self._types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            if "thinking" in str(exc).lower() and "thinking_config" in config_kwargs:
                config_kwargs.pop("thinking_config")
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=self._types.GenerateContentConfig(**config_kwargs),
                )
            else:
                raise

        text = getattr(response, "text", None)
        if not text:
            raise TranslationError("Gemini returned an empty response.")
        return text


class OpenAICompatibleLLMClient:
    """OpenAI / DeepSeek / OpenRouter / Ollama-compatible translation client."""

    name = "openai"

    def __init__(
        self,
        api_key: str | None,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str | None = None,
    ) -> None:
        resolved_key = api_key
        if not resolved_key:
            if base_url and ("localhost" in base_url or "127.0.0.1" in base_url):
                resolved_key = "local-no-key"
            else:
                raise TranslationError(
                    "OpenAI API key is missing. Set OPENAI_API_KEY in .env or pass --api-key."
                )

        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model
        self.base_url = base_url

    def _is_official_openai(self) -> bool:
        return not self.base_url or "openai.com" in self.base_url.lower()

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        # Disable hidden reasoning tokens on GPT-6 Luna so subtitle translation stays fast and lowest-cost.
        if self._is_official_openai() and self.model.lower().startswith("gpt-6-luna"):
            params["reasoning_effort"] = "none"

        # Retry without optional parameters if a specific model/provider rejects them.
        response = None
        for _ in range(4):
            try:
                response = self._client.chat.completions.create(**params)
                break
            except Exception as exc:
                err_lower = str(exc).lower()
                if "response_format" in err_lower and "response_format" in params:
                    params.pop("response_format")
                elif "temperature" in err_lower and "temperature" in params:
                    params.pop("temperature")
                elif "reasoning_effort" in err_lower and "reasoning_effort" in params:
                    params.pop("reasoning_effort")
                else:
                    raise

        content = response.choices[0].message.content if (response and response.choices) else None
        if not content:
            raise TranslationError("LLM provider returned an empty response.")
        return content


def create_llm_client(
    provider: TranslatorName = "openai",
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """Create a configured LLM translation client using overrides or environment settings."""
    settings = load_settings()

    if provider == "gemini":
        resolved_key = api_key or settings.gemini_api_key or ""
        resolved_model = model or settings.default_gemini_model
        return GeminiLLMClient(api_key=resolved_key, model=resolved_model)

    if provider == "openai":
        resolved_key = api_key or settings.openai_api_key
        resolved_base = base_url or settings.openai_base_url
        resolved_model = model or settings.default_openai_model
        return OpenAICompatibleLLMClient(
            api_key=resolved_key,
            model=resolved_model,
            base_url=resolved_base,
        )

    raise TranslationError(f"Unsupported translator provider: {provider}")


def build_system_prompt(source_language: str, context_prompt: str = "") -> str:
    """Build a tailored subtitle translation system prompt for Simplified Chinese output."""
    lang_code = source_language.lower()
    if lang_code.startswith("ja"):
        lang_label = "Japanese (日本語)"
        lang_tips = (
            "- Pay close attention to Japanese context, omitted subjects (主语省略), honorifics (敬语/昵称), "
            "and conversational fillers so the Chinese reads naturally like a professional fansub/localization.\n"
            "- Translate idioms, anime/drama expressions, and sentence-ending particles by conveying tone "
            "rather than translating literally.\n"
        )
    elif lang_code.startswith("en"):
        lang_label = "English"
        lang_tips = (
            "- Translate spoken English idioms, slang, technical terms, and phrasal verbs into natural, "
            "idiomatic Simplified Chinese rather than stiff word-for-word translation.\n"
            "- Keep sentence structure smooth across split subtitle cues when a single English sentence spans "
            "multiple consecutive IDs.\n"
        )
    else:
        lang_label = f"spoken video audio ({source_language})"
        lang_tips = "- Translate naturally into idiomatic Simplified Chinese suitable for video subtitles.\n"

    extra_context = ""
    if context_prompt.strip():
        extra_context = f"\nAdditional domain context / glossary provided by user:\n{context_prompt.strip()}\n"

    return (
        f"You are an expert audiovisual subtitle translator specializing in translating {lang_label} "
        "into Simplified Chinese (简体中文).\n\n"
        "Translation rules:\n"
        "1. Always output Simplified Chinese (简体中文) using natural Chinese punctuation (avoid unnecessary trailing periods on short single-clause subtitles; keep question/exclamation marks when appropriate).\n"
        f"{lang_tips}"
        "2. Keep each subtitle concise and easy to read on screen while preserving full meaning and nuance.\n"
        "3. Use the provided `context_before` entries ONLY as reference for continuity (pronouns, tone, names). "
        "Do NOT include `context_before` items in your output.\n"
        "4. You MUST return a translation for EVERY item in `items`, preserving each item's exact integer `id` "
        "(1-to-1 alignment; never merge, skip, or renumber IDs).\n"
        "5. Respond strictly with valid JSON matching this exact schema:\n"
        '{"translations": [{"id": 1, "text": "简体中文字幕"}]}\n'
        f"{extra_context}"
    )


def parse_translation_response(raw_text: str) -> dict[int, str]:
    """Parse JSON returned by an LLM into a mapping of `{segment_id: translated_text}`."""
    cleaned = _CODE_FENCE_RE.sub("", raw_text.strip()).strip()

    # Fallback: extract first JSON object/array if extra prose leaked around it.
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        obj_start = cleaned.find("{")
        obj_end = cleaned.rfind("}")
        arr_start = cleaned.find("[")
        arr_end = cleaned.rfind("]")
        if obj_start != -1 and obj_end > obj_start:
            payload = json.loads(cleaned[obj_start : obj_end + 1])
        elif arr_start != -1 and arr_end > arr_start:
            payload = json.loads(cleaned[arr_start : arr_end + 1])
        else:
            raise TranslationError(f"LLM did not return valid JSON: {cleaned[:200]}")

    result: dict[int, str] = {}

    if isinstance(payload, dict):
        entries = payload.get("translations")
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, dict) and "id" in item and "text" in item:
                    try:
                        result[int(item["id"])] = str(item["text"]).strip()
                    except (TypeError, ValueError):
                        continue
            return result

        # Fallback format: {"1": "译文", "2": "译文"}
        for key, value in payload.items():
            try:
                seg_id = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, str):
                result[seg_id] = value.strip()
        return result

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "id" in item and "text" in item:
                try:
                    result[int(item["id"])] = str(item["text"]).strip()
                except (TypeError, ValueError):
                    continue
        return result

    raise TranslationError("Unexpected JSON structure returned by LLM.")


class SubtitleTranslator:
    """Batch subtitle translator with sliding context window and retry resilience."""

    def __init__(
        self,
        client: LLMClient,
        *,
        batch_size: int = 30,
        context_window: int = 5,
        max_retries: int = 3,
        retry_base_delay: float = 1.5,
    ) -> None:
        self.client = client
        self.batch_size = max(1, batch_size)
        self.context_window = max(0, context_window)
        self.max_retries = max(1, max_retries)
        self.retry_base_delay = retry_base_delay

    def _call_batch(
        self,
        system_prompt: str,
        batch: list[SubtitleSegment],
        context_before: list[SubtitleSegment],
    ) -> dict[int, str]:
        user_payload = {
            "context_before": [
                {
                    "id": seg.index,
                    "source": seg.source_text,
                    "zh": seg.translated_text,
                }
                for seg in context_before
            ],
            "items": [{"id": seg.index, "source": seg.source_text} for seg in batch],
        }
        user_prompt = json.dumps(user_payload, ensure_ascii=False)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.client.complete_json(system_prompt, user_prompt)
                parsed = parse_translation_response(raw)
                if parsed:
                    return parsed
                raise TranslationError("Parsed translation mapping was empty.")
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))

        raise TranslationError(f"Batch translation failed after {self.max_retries} attempts: {last_error}")

    def translate_segments(
        self,
        segments: list[SubtitleSegment],
        source_language: str,
        *,
        context_prompt: str = "",
        on_progress: ProgressCallback | None = None,
    ) -> list[SubtitleSegment]:
        """Translate all segments into Simplified Chinese while maintaining cross-batch context."""
        if not segments:
            return []

        system_prompt = build_system_prompt(source_language, context_prompt=context_prompt)
        translated: list[SubtitleSegment] = []
        total = len(segments)
        total_batches = (total + self.batch_size - 1) // self.batch_size

        for batch_idx, start in enumerate(range(0, total, self.batch_size), start=1):
            batch = segments[start : start + self.batch_size]
            context_before = translated[-self.context_window :] if self.context_window > 0 else []

            if on_progress:
                pct = ((batch_idx - 1) / total_batches) * 100.0
                on_progress(
                    pct,
                    f"Translating batch {batch_idx}/{total_batches} ({len(batch)} lines) via {self.client.name}/{self.client.model}...",
                )

            mapping = self._call_batch(system_prompt, batch, context_before)

            # Retry any missing IDs once in a targeted sub-batch.
            missing = [seg for seg in batch if seg.index not in mapping or not mapping[seg.index]]
            if missing and len(missing) < len(batch):
                retry_map = self._call_batch(system_prompt, missing, context_before)
                mapping.update({k: v for k, v in retry_map.items() if v})

            for seg in batch:
                zh_text = mapping.get(seg.index, "").strip() or seg.source_text
                translated.append(seg.model_copy(update={"translated_text": zh_text}))

            if on_progress:
                pct = (batch_idx / total_batches) * 100.0
                on_progress(pct, f"Translated {len(translated)}/{total} subtitle cues")

        return translated
