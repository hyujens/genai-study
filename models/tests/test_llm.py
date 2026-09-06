from collections.abc import Iterator
from typing import Any, cast

from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from models import _llm as llm
from models._config import LLMConfig


def make_completion(message: dict[str, Any], finish_reason: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "logprobs": None,
                    "message": message,
                }
            ],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
        }
    )


def make_chunk(delta: dict[str, Any]) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "delta": delta,
                    "finish_reason": None,
                    "index": 0,
                    "logprobs": None,
                }
            ],
            "created": 0,
            "model": "test-model",
            "object": "chat.completion.chunk",
        }
    )


def test_process_response_returns_text():
    service = llm.Service(LLMConfig(model="test-model"))
    completion = make_completion(
        {"role": "assistant", "content": "hello"},
        finish_reason="stop",
    )

    assert list(service._process_response(completion)) == [llm.ChatResponse("hello")]


def test_process_response_returns_complete_tool_calls():
    service = llm.Service(LLMConfig(model="test-model"))
    completion = make_completion(
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "get_current_time",
                        "arguments": "{}",
                    },
                }
            ],
        },
        finish_reason="tool_calls",
    )

    assert list(service._process_response(completion)) == [
        llm.ToolCallsResponse(
            content="checking",
            tools=(llm.ToolCall("call-1", "get_current_time", "{}"),),
        )
    ]


def test_process_stream_response_merges_interleaved_tool_call_fragments():
    service = llm.Service(LLMConfig(model="test-model"))
    chunks = [
        make_chunk({"content": "checking "}),
        make_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "first_tool",
                            "arguments": '{"x":',
                        },
                    },
                    {
                        "index": 1,
                        "id": "call-2",
                        "type": "function",
                        "function": {
                            "name": "second_tool",
                            "arguments": '{"y":',
                        },
                    },
                ]
            }
        ),
        make_chunk(
            {
                "tool_calls": [
                    {"index": 1, "function": {"arguments": "2}"}},
                    {"index": 0, "function": {"arguments": "1}"}},
                ]
            }
        ),
    ]

    events = list(
        service._process_stream_response(
            cast(Stream[ChatCompletionChunk], cast(object, iter(chunks)))
        )
    )

    assert events == [
        llm.ChatResponse("checking "),
        llm.ToolCallsResponse(
            content="",
            tools=(
                llm.ToolCall("call-1", "first_tool", '{"x":1}'),
                llm.ToolCall("call-2", "second_tool", '{"y":2}'),
            ),
        ),
    ]


def test_do_inference_serializes_all_message_roles(monkeypatch):
    service = llm.Service(LLMConfig(model="test-model"))
    completion = make_completion(
        {"role": "assistant", "content": "done"},
        finish_reason="stop",
    )
    captured: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> ChatCompletion:
        captured.update(kwargs)
        return completion

    monkeypatch.setattr(service._client.chat.completions, "create", fake_create)
    messages = [
        llm.Message(llm.OpenAIStyleRole.System, "system"),
        llm.Message(llm.OpenAIStyleRole.User, "question"),
        llm.Message(
            llm.OpenAIStyleRole.Assistant,
            "",
            tool_calls=[llm.ToolCall("call-1", "clock", "{}")],
        ),
        llm.Message(
            llm.OpenAIStyleRole.Tool,
            '{"time":"now"}',
            tool_call_id="call-1",
        ),
    ]

    result: Iterator[llm.InferenceResponse] = service.do_inference(
        messages,
        stream=False,
    )

    assert list(result) == [llm.ChatResponse("done")]
    assert captured["stream"] is False
    assert captured["messages"] == [
        {"content": "system", "role": "system"},
        {"content": "question", "role": "user"},
        {
            "content": "",
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"arguments": "{}", "name": "clock"},
                    "type": "function",
                }
            ],
        },
        {"content": '{"time":"now"}', "role": "tool", "tool_call_id": "call-1"},
    ]


def test_do_inference_dispatches_stream_response(monkeypatch):
    service = llm.Service(LLMConfig(model="test-model"))

    class FakeStream:
        def __iter__(self) -> Iterator[ChatCompletionChunk]:
            return iter([make_chunk({"content": "streamed"})])

    monkeypatch.setattr(llm, "Stream", FakeStream)
    monkeypatch.setattr(
        service._client.chat.completions,
        "create",
        lambda **kwargs: FakeStream(),
    )

    assert list(service.do_inference([], stream=True)) == [llm.ChatResponse("streamed")]
