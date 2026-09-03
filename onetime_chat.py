import argparse
import sys

import requests

from models import llm
from models.config import DEFAULT_INFERENCE_ENDPOINT, LLMConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message", type=str, help="chat with ai")
    args = parser.parse_args()

    resp = requests.get(f"{DEFAULT_INFERENCE_ENDPOINT}/v1/models")
    if resp.status_code != 200:
        print("failed to get model list: ", resp.status_code)
        sys.exit(-1)

    model = resp.json()["models"][0]["model"]

    svc = llm.Service(LLMConfig(model=model))
    answer = svc.inference_stream(
        [
            llm.Message(
                llm.OpenAIStyleRole.System,
                """
        You are friendly and smart and willing to answer any what you've learnt before.
        If you do not know answers from your knowledge, you will kindly tell the user you do not know.
        Although you are smart, it is possible to get insufficient information. If you really get it, 
        please raise more questions to clarify it with the user.
        """,
            ),
            llm.Message(llm.OpenAIStyleRole.User, args.message),
        ]
    )

    for text in answer:
        print(text, end="", flush=True)


if __name__ == "__main__":
    main()
