from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from openai import OpenAI, Stream, omit
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolUnionParam,
    ChatCompletionUserMessageParam,
)

from ._config import LLMConfig


class OpenAIStyleRole(StrEnum):
    System = "system"
    User = "user"
    Assistant = "assistant"
    Tool = "tool"


@dataclass
class Message:
    role: str
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ChatResponse:
    chunk: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolCallsResponse:
    content: str
    tools: tuple[ToolCall, ...]


InferenceResponse = ChatResponse | ToolCallsResponse


class Service:
    def __init__(self, conf: LLMConfig) -> None:
        self._client = OpenAI(base_url=conf.endpoint, api_key=conf.api_key)
        self._model = conf.model
        self._tools = conf.tools

    def _process_response(
        self, completion: ChatCompletion
    ) -> Iterator[InferenceResponse]:
        if not completion.choices:
            raise RuntimeError("empty choices from server")

        content = ""
        message = completion.choices[0].message
        if message.content:
            content = message.content

        if message.tool_calls:
            tools: list[ToolCall] = []
            for tool in message.tool_calls:
                if tool.type != "function":
                    continue
                tools.append(
                    ToolCall(
                        id=tool.id,
                        name=tool.function.name,
                        arguments=tool.function.arguments,
                    )
                )

            yield ToolCallsResponse(content=content, tools=tuple(tools))
            return

        yield ChatResponse(content)
        return

    def _process_stream_response(
        self, completion: Stream[ChatCompletionChunk]
    ) -> Iterator[InferenceResponse]:
        stream_tool_calls: dict[int, dict[str, list[str]]] = {}

        for stream in completion:
            if not stream.choices:
                continue

            chunk = stream.choices[0].delta
            if chunk.content:
                yield ChatResponse(chunk.content)

            for chunk_tool in chunk.tool_calls or []:
                buffer = stream_tool_calls.setdefault(
                    chunk_tool.index, {"id": [], "name": [], "arguments": []}
                )

                if chunk_tool.id:
                    buffer["id"].append(chunk_tool.id)

                if chunk_tool.function:
                    if chunk_tool.function.name:
                        buffer["name"].append(chunk_tool.function.name)
                    if chunk_tool.function.arguments:
                        buffer["arguments"].append(chunk_tool.function.arguments)

        if stream_tool_calls:
            tools: list[ToolCall] = [
                ToolCall(
                    id="".join(buf["id"]),
                    name="".join(buf["name"]),
                    arguments="".join(buf["arguments"]),
                )
                for _, buf in sorted(stream_tool_calls.items())
            ]

            yield ToolCallsResponse(content="", tools=tuple(tools))

    def do_inference(
        self, messages: list[Message], stream=True
    ) -> Iterator[InferenceResponse]:
        chat_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            match msg.role:
                case "system":
                    chat_messages.append(
                        ChatCompletionSystemMessageParam(
                            content=msg.content, role=msg.role
                        )
                    )
                case "user":
                    chat_messages.append(
                        ChatCompletionUserMessageParam(
                            content=msg.content, role=msg.role
                        )
                    )
                case "assistant":
                    chat_messages.append(
                        ChatCompletionAssistantMessageParam(
                            content=msg.content,
                            role=msg.role,
                            tool_calls=[]
                            if msg.tool_calls is None
                            else [
                                ChatCompletionMessageFunctionToolCallParam(
                                    id=param.id,
                                    function={
                                        "arguments": param.arguments,
                                        "name": param.name,
                                    },
                                    type="function",
                                )
                                for param in msg.tool_calls
                            ],
                        )
                    )
                case "tool":
                    if msg.tool_call_id is None:
                        raise RuntimeError("tool_call_id should not be empty")

                    chat_messages.append(
                        ChatCompletionToolMessageParam(
                            content=msg.content,
                            role=msg.role,
                            tool_call_id=msg.tool_call_id,
                        )
                    )
        completion: ChatCompletion | Stream[ChatCompletionChunk] = (
            self._client.chat.completions.create(
                model=self._model,
                messages=chat_messages,
                tools=[
                    cast(ChatCompletionToolUnionParam, tool.json_schema())
                    for tool in self._tools
                ]
                if self._tools
                else omit,
                stream=stream,
            )
        )

        if isinstance(completion, ChatCompletion):
            return self._process_response(completion)
        if isinstance(completion, Stream):
            return self._process_stream_response(completion)

        raise RuntimeError("unknown chat completion type")

    def inference(self, messages: list[Message]) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                cast(
                    ChatCompletionMessageParam,
                    {
                        "role": msg.role,
                        "content": msg.content,
                    },
                )
                for msg in messages
            ],
            tools=[
                cast(ChatCompletionToolUnionParam, tool.json_schema())
                for tool in self._tools
            ]
            if self._tools
            else omit,
        )
        if not completion.choices:
            raise RuntimeError("empty choices from server")

        message = completion.choices[0].message
        if message is None:
            raise RuntimeError("Abnormal behavior: choice.message is none")

        if message.content is None:
            raise NotImplementedError("non-text llm is not implemented yet")

        return message.content

    def inference_stream(self, messages: list[Message]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[
                cast(
                    ChatCompletionMessageParam,
                    {
                        "role": msg.role,
                        "content": msg.content,
                    },
                )
                for msg in messages
            ],
            tools=[
                cast(ChatCompletionToolUnionParam, tool.json_schema())
                for tool in self._tools
            ]
            if self._tools
            else omit,
            stream=True,
        )

        for chunk in stream:
            if len(chunk.choices) == 0:
                raise RuntimeError("Abnormal behavior: choice.message is none")

            content = chunk.choices[0].delta.content
            if content is not None:
                yield content
