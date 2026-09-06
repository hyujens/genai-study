import json
from collections.abc import Iterator
from dataclasses import asdict
from datetime import UTC, datetime

import requests
from models import (
    DEFAULT_INFERENCE_ENDPOINT,
    ChatMessage,
    ChatResponse,
    LLMConfig,
    LLMModel,
    OpenAIStyleRole,
    ToolCall,
    ToolCallsResponse,
    ToolFuncDef,
)


class Agent:
    def __init__(
        self, role_description: str, max_reasoning_steps=10, stream=True
    ) -> None:
        self._system_role = ChatMessage(OpenAIStyleRole.System, role_description)
        self._trajectory: list[ChatMessage] = []

        resp = requests.get(f"{DEFAULT_INFERENCE_ENDPOINT}/v1/models")
        if resp.status_code != 200:
            raise RuntimeError("failed to get models: ", resp.status_code)

        self._model = resp.json()["models"][0]["model"]
        self._llm_service = LLMModel(
            LLMConfig(
                model=self._model,
                tools=[
                    ToolFuncDef(
                        name="get_current_time",
                        description='call this tool when current time is needed. The returned data will be {"datetime": "2026-09-06 15:05:43.016975"}',
                        parameters=[],
                    )
                ],
            )
        )
        self._tools = {
            "get_current_time": lambda id: {
                "tool_call_id": id,
                "datetime": str(datetime.now(UTC)),
            }
        }

        self._max_reasoning_steps = max_reasoning_steps
        self._stream = stream

    def get_model(self) -> str:
        return self._model

    def get_trajactory(self) -> list[dict]:
        return [asdict(his) for his in self._trajectory]

    def _reasoning_inference(self, step_th: int) -> Iterator[str]:
        if step_th >= self._max_reasoning_steps:
            yield "too many steps to find out answer! Stop!"
            return
        iter = self._llm_service.do_inference(
            [self._system_role, *self._trajectory], stream=self._stream
        )

        chunks: list[str] = []
        tools: list[ToolCall] = []
        for words in iter:
            if type(words) is ChatResponse:
                yield words.chunk
                chunks.append(words.chunk)
            if type(words) is ToolCallsResponse:
                tools.extend(words.tools)

        self._trajectory.append(
            ChatMessage(
                role=OpenAIStyleRole.Assistant,
                content="".join(chunks),
                tool_calls=tools if tools else None,
            )
        )

        if tools:
            for tool in tools:
                if tool.name not in self._tools:
                    raise RuntimeError(f"no {tool.name} found")

                self._trajectory.append(
                    ChatMessage(
                        role=OpenAIStyleRole.Tool,
                        tool_call_id=tool.id,
                        content=json.dumps(self._tools[tool.name](tool.id)),
                    )
                )
            yield from self._reasoning_inference(step_th + 1)
            return

    def inference(self, message: str) -> Iterator[str]:
        self._trajectory.append(ChatMessage(OpenAIStyleRole.User, message))

        yield from self._reasoning_inference(0)
        return
