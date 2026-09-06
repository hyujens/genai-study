import argparse

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from agent import ChatAgent, SystemRole


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message", type=str, help="chat with ai")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    llmchat = ChatAgent(SystemRole.Stranger)
    answer = llmchat.inference(args.message)

    console = Console()
    text = ""
    with Live(Markdown(""), console=console, refresh_per_second=3) as live:
        for chunk in answer:
            if chunk:
                text += chunk
                live.update(Markdown(text))

    if args.debug:
        print(f"\n\n\tmodel: {llmchat.get_model()}")
        print("\ttractory: ")
        for his in llmchat.get_trajactory():
            role = his["role"]
            conent = his["content"]
            ellipsis = "" if len(conent) < 50 else "..."
            print(f"\t\t{role}: {his['content'][:50]} {ellipsis}")


if __name__ == "__main__":
    main()
