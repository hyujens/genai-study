import argparse

from agent.role import SystemRole

from agent import chat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message", type=str, help="chat with ai")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    llmchat = chat.Agent(SystemRole.Stranger)
    answer = llmchat.inference(args.message)

    for text in answer:
        print(text, end="", flush=True)
    print("\n")

    if args.debug:
        print(f"model: {llmchat.get_model()}")
        print("tractory: ")
        print(llmchat.get_trajactory())


if __name__ == "__main__":
    main()
