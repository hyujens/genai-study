"""LLM model service package."""

from ._config import DEFAULT_INFERENCE_ENDPOINT, LLMConfig
from ._llm import (
    ChatResponse,
    InferenceResponse,
    OpenAIStyleRole,
    ToolCall,
    ToolCallsResponse,
)
from ._llm import Message as ChatMessage
from ._llm import Service as LLMModel
from ._tool import FuncDef as ToolFuncDef
from ._tool import ParametertDef as ToolFuncParamDef

__all__ = [
    "DEFAULT_INFERENCE_ENDPOINT",
    "ChatMessage",
    "ChatResponse",
    "InferenceResponse",
    "LLMConfig",
    "LLMModel",
    "OpenAIStyleRole",
    "ToolCall",
    "ToolCallsResponse",
    "ToolFuncDef",
    "ToolFuncParamDef",
]
