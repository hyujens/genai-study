from collections.abc import Iterator

import pytest
from agent._chat import Agent
from models import _llm as llm


class FakeLLMService:
    def __init__(self, responses: list[list[llm.InferenceResponse]]) -> None:
        self.responses = iter(responses)
        self.stream_values: list[bool] = []
        self.messages: list[list[llm.Message]] = []

    def do_inference(
        self,
        messages: list[llm.Message],
        stream: bool = True,
    ) -> Iterator[llm.InferenceResponse]:
        self.messages.append(messages)
        self.stream_values.append(stream)
        return iter(next(self.responses))


def make_agent(
    service: FakeLLMService,
    *,
    stream: bool,
    max_reasoning_steps: int = 10,
) -> Agent:
    agent = Agent.__new__(Agent)
    agent._system_role = llm.Message(llm.OpenAIStyleRole.System, "system")
    agent._trajectory = []
    agent._llm_service = service
    agent._tools = {}
    agent._max_reasoning_steps = max_reasoning_steps
    agent._stream = stream
    return agent


@pytest.mark.parametrize("stream", [True, False])
def test_inference_forwards_stream_and_records_text_response(stream: bool):
    service = FakeLLMService([[llm.ChatResponse("answer")]])
    agent = make_agent(service, stream=stream)

    assert list(agent.inference("question")) == ["answer"]
    assert service.stream_values == [stream]
    assert agent._trajectory == [
        llm.Message(llm.OpenAIStyleRole.User, "question"),
        llm.Message(llm.OpenAIStyleRole.Assistant, "answer"),
    ]


@pytest.mark.parametrize("stream", [True, False])
def test_inference_executes_tool_and_continues_to_final_response(stream: bool):
    tool_call = llm.ToolCall("call-1", "clock", "{}")
    service = FakeLLMService(
        [
            [llm.ToolCallsResponse("ignored content", (tool_call,))],
            [llm.ChatResponse("final answer")],
        ]
    )
    agent = make_agent(service, stream=stream)
    agent._tools["clock"] = lambda tool_call_id: {
        "tool_call_id": tool_call_id,
        "time": "now",
    }

    assert list(agent.inference("question")) == ["final answer"]
    assert service.stream_values == [stream, stream]
    assert [message.role for message in agent._trajectory] == [
        llm.OpenAIStyleRole.User,
        llm.OpenAIStyleRole.Assistant,
        llm.OpenAIStyleRole.Tool,
        llm.OpenAIStyleRole.Assistant,
    ]
    assert agent._trajectory[1].tool_calls == [tool_call]
    assert agent._trajectory[1].content == ""
    assert agent._trajectory[2].tool_call_id == "call-1"
    assert agent._trajectory[3].content == "final answer"


def test_inference_stops_before_exceeding_max_reasoning_steps():
    service = FakeLLMService([])
    agent = make_agent(service, stream=True, max_reasoning_steps=0)

    assert list(agent.inference("question")) == [
        "too many steps to find out answer! Stop!"
    ]
    assert service.stream_values == []


def test_inference_rejects_unknown_tool():
    service = FakeLLMService(
        [
            [
                llm.ToolCallsResponse(
                    "",
                    (llm.ToolCall("call-1", "unknown", "{}"),),
                )
            ]
        ]
    )
    agent = make_agent(service, stream=True)

    with pytest.raises(RuntimeError, match="no unknown found"):
        list(agent.inference("question"))
