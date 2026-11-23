"""Tests for LLM batch translation, JSON parsing, context windowing, and retry recovery."""

import json

from subgen.models import SubtitleSegment
from subgen.translate import SubtitleTranslator, parse_translation_response


class FakeLLMClient:
    name = "fake"
    model = "fake-v1"

    def __init__(self, responses: list[str] | None = None) -> None:
        self.calls: list[dict] = []
        self._responses = list(responses or [])

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt)
        self.calls.append({"system": system_prompt, "user": payload})
        if self._responses:
            return self._responses.pop(0)
        return json.dumps(
            {
                "translations": [
                    {"id": item["id"], "text": f"中文:{item['source']}"}
                    for item in payload["items"]
                ]
            },
            ensure_ascii=False,
        )


def test_parse_translation_response_formats():
    # Standard schema wrapped in markdown fence
    raw1 = '```json\n{"translations": [{"id": 1, "text": "你好"}, {"id": 2, "text": "世界"}]}\n```'
    assert parse_translation_response(raw1) == {1: "你好", 2: "世界"}

    # Key-value fallback
    raw2 = '{"1": "早上好", "2": "晚安"}'
    assert parse_translation_response(raw2) == {1: "早上好", 2: "晚安"}


def test_batching_passes_sliding_context_window():
    segments = [
        SubtitleSegment(index=i, start=float(i), end=float(i + 1), source_text=f"Line {i}")
        for i in range(1, 6)
    ]
    client = FakeLLMClient()
    translator = SubtitleTranslator(client, batch_size=2, context_window=2)

    out = translator.translate_segments(segments, source_language="ja", context_prompt="Anime glossary")

    assert len(out) == 5
    assert [s.translated_text for s in out] == [f"中文:Line {i}" for i in range(1, 6)]
    assert len(client.calls) == 3

    # First batch has no prior context.
    assert client.calls[0]["user"]["context_before"] == []
    # Second batch sees translated cues 1 & 2.
    assert [c["id"] for c in client.calls[1]["user"]["context_before"]] == [1, 2]
    assert client.calls[1]["user"]["context_before"][0]["zh"] == "中文:Line 1"
    # Third batch sees translated cues 3 & 4.
    assert [c["id"] for c in client.calls[2]["user"]["context_before"]] == [3, 4]
    assert "Anime glossary" in client.calls[0]["system"]


def test_missing_id_triggers_targeted_retry():
    segments = [
        SubtitleSegment(index=1, start=0.0, end=1.0, source_text="One"),
        SubtitleSegment(index=2, start=1.0, end=2.0, source_text="Two"),
        SubtitleSegment(index=3, start=2.0, end=3.0, source_text="Three"),
    ]
    # First call omits ID 2; retry fills ID 2.
    client = FakeLLMClient(
        responses=[
            json.dumps({"translations": [{"id": 1, "text": "一"}, {"id": 3, "text": "三"}]}),
            json.dumps({"translations": [{"id": 2, "text": "二"}]}),
        ]
    )
    translator = SubtitleTranslator(client, batch_size=3, context_window=0)
    out = translator.translate_segments(segments, source_language="en")

    assert [s.translated_text for s in out] == ["一", "二", "三"]
    assert len(client.calls) == 2
    assert [item["id"] for item in client.calls[1]["user"]["items"]] == [2]


def test_openai_client_sets_reasoning_effort_none_for_gpt6_luna():
    from types import SimpleNamespace
    from subgen.translate import OpenAICompatibleLLMClient

    captured_kwargs: list[dict] = []

    def fake_create(**kwargs):
        captured_kwargs.append(dict(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"translations": [{"id": 1, "text": "你好"}]}'))]
        )

    client = OpenAICompatibleLLMClient(api_key="sk-test", model="gpt-6-luna")
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

    result = client.complete_json("sys", "user")
    assert '"你好"' in result
    assert captured_kwargs[0]["model"] == "gpt-6-luna"
    assert captured_kwargs[0]["reasoning_effort"] == "none"


def test_openai_client_falls_back_when_temperature_unsupported():
    from types import SimpleNamespace
    from subgen.translate import OpenAICompatibleLLMClient

    calls: list[dict] = []

    def fake_create(**kwargs):
        calls.append(dict(kwargs))
        if "temperature" in kwargs:
            raise ValueError("Unsupported parameter: 'temperature' is not supported with this model.")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"translations": [{"id": 1, "text": "世界"}]}'))]
        )

    client = OpenAICompatibleLLMClient(api_key="sk-test", model="gpt-5-nano")
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

    result = client.complete_json("sys", "user")
    assert '"世界"' in result
    assert len(calls) == 2
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]

