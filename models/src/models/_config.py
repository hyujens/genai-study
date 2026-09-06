from dataclasses import dataclass

from models._tool import FuncDef

DEFAULT_INFERENCE_ENDPOINT = "http://localhost:8080"


@dataclass
class LLMConfig:
    model: str
    endpoint: str = DEFAULT_INFERENCE_ENDPOINT
    api_key: str = "local"
    tools: list[FuncDef] | None = None
