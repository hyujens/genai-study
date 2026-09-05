from collections.abc import Iterator

import requests
from models import llm
from models.config import DEFAULT_INFERENCE_ENDPOINT, LLMConfig


class Agent:
    def __init__(self, role_description: str) -> None:
        self.system_role = llm.Message(llm.OpenAIStyleRole.System, role_description)
        self.trajectory: list[llm.Message] = []

        resp = requests.get(f"{DEFAULT_INFERENCE_ENDPOINT}/v1/models")
        if resp.status_code != 200:
            raise RuntimeError("failed to get models: ", resp.status_code)

        self.model = resp.json()["models"][0]["model"]
        self.llm_service = llm.Service(LLMConfig(model=self.model))

    def get_model(self) -> str:
        return self.model

    def inference(self, message: str) -> Iterator[str]:
        self.trajectory.append(llm.Message(llm.OpenAIStyleRole.User, message))
        iter = self.llm_service.inference_stream([self.system_role, *self.trajectory])

        chunks: list[str] = []
        for words in iter:
            yield words
            chunks.append(words)

        self.trajectory.append(
            llm.Message(llm.OpenAIStyleRole.Assistant, "".join(chunks))
        )
