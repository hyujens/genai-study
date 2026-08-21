from openai import OpenAI

from .server import infernece_host

MODEL_0_6_B = "Qwen/Qwen3-0.6B-GGUF"


def onetime_chat(message: str) -> str:
    client = OpenAI(base_url=infernece_host, api_key="local")

    response = client.chat.completions.create(
        model=MODEL_0_6_B,
        messages=[
            {"role": "system", "content": "You are a kind assistant."},
            {"role": "user", "content": message},
        ],
    )

    result: str | None = (
        "no response"
        if len(response.choices) == 0
        else response.choices[0].message.content
    )

    return "no content" if result is None else result
