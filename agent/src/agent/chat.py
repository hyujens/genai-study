import json
from collections.abc import Iterator
from dataclasses import asdict
from datetime import UTC, datetime

import requests
from models import llm, tool
from models.config import DEFAULT_INFERENCE_ENDPOINT, LLMConfig


class Agent:
    def __init__(
        self, role_description: str, max_reasoning_steps=10, stream=True
    ) -> None:
        self.system_role = llm.Message(llm.OpenAIStyleRole.System, role_description)
        self.trajectory: list[llm.Message] = []

        resp = requests.get(f"{DEFAULT_INFERENCE_ENDPOINT}/v1/models")
        if resp.status_code != 200:
            raise RuntimeError("failed to get models: ", resp.status_code)

        self.model = resp.json()["models"][0]["model"]
        self.llm_service = llm.Service(
            LLMConfig(
                model=self.model,
                tools=[
                    tool.FuncDef(
                        name="get_current_time",
                        description='call this tool when current time is needed. The returned data will be {"datetime": "2026-09-06 15:05:43.016975"}',
                        parameters=[],
                    )
                ],
            )
        )
        self.tools = {
            "get_current_time": lambda id: {
                "tool_call_id": id,
                "datetime": str(datetime.now(UTC)),
            }
        }

        self.max_reasoning_steps = max_reasoning_steps
        self.stream = stream

    def get_model(self) -> str:
        return self.model

    def get_trajactory(self) -> list[dict]:
        return [asdict(his) for his in self.trajectory]

    def reasoning_inference(self, step_th: int) -> Iterator[str]:
        if step_th >= self.max_reasoning_steps:
            yield "too many steps to find out answer! Stop!"
            return
        iter = self.llm_service.do_inference(
            [self.system_role, *self.trajectory], stream=self.stream
        )

        chunks: list[str] = []
        tools: list[llm.ToolCall] = []
        for words in iter:
            if type(words) is llm.ChatResponse:
                yield words.chunk
                chunks.append(words.chunk)
            if type(words) is llm.ToolCallsResponse:
                tools.extend(words.tools)

        self.trajectory.append(
            llm.Message(
                role=llm.OpenAIStyleRole.Assistant,
                content="".join(chunks),
                tool_calls=tools if tools else None,
            )
        )

        if tools:
            for tool in tools:
                if tool.name not in self.tools:
                    raise RuntimeError(f"no {tool.name} found")

                self.trajectory.append(
                    llm.Message(
                        role=llm.OpenAIStyleRole.Tool,
                        tool_call_id=tool.id,
                        content=json.dumps(self.tools[tool.name](tool.id)),
                    )
                )
            yield from self.reasoning_inference(step_th + 1)
            return

    def inference(self, message: str) -> Iterator[str]:
        self.trajectory.append(llm.Message(llm.OpenAIStyleRole.User, message))

        yield from self.reasoning_inference(0)
        return
