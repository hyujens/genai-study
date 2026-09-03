from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from .config import LLMConfig
from collections.abc import Iterator


class OpenAIStyleRole(StrEnum):
    System = "system"
    User = "user"
    Assistant = "assistant"


@dataclass
class Message:
    role: str
    content: str


class Service:
    def __init__(self, conf: LLMConfig) -> None:
        self.client = OpenAI(base_url=conf.endpoint, api_key=conf.api_key)
        self.model = conf.model

    def inference(self, messages: list[Message]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
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
        stream = self.client.chat.completions.create(
            model=self.model,
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
            stream=True,
        )

        for chunk in stream:
            if len(chunk.choices) == 0:
                raise RuntimeError("Abnormal behavior: choice.message is none")

            content = chunk.choices[0].delta.content
            if content is not None:
                yield content
