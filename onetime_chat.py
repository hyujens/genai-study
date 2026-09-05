import argparse

from agent.role import SystemRole

from agent import chat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message", type=str, help="chat with ai")
    args = parser.parse_args()

    answer = chat.Agent(SystemRole.Stranger).inference(args.message)

    for text in answer:
        print(text, end="", flush=True)


if __name__ == "__main__":
    main()
