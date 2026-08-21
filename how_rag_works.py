from models import qwen
from rag.concept_with_keyword_search import knowledge, naive_keyword_search


def synthesize_user_question(query: str, documents: list[str]) -> str:
    context = "\n\n".join(documents)
    return f"""
    Answer the question based on the following context:

    Context:
    {context}

    Question: {query}

    Answer:
    """


def main():
    user_messages = [
        "What is Underwhelming Spatula?",
        "Who wrote Dubious Parenting Tips?",
        "What is the weather today?",
    ]

    for msg in user_messages:
        print(f"user: {msg}")
        response = ""
        # Retreival
        answers = naive_keyword_search(msg, knowledge)
        print(f"(rag: {answers})")

        # Generate answer
        if len(answers) == 0:
            response = "I don't have enough information to answer that question."
        else:
            response = qwen.onetime_chat(synthesize_user_question(msg, answers))

        print(f"assist: {response}")


if __name__ == "__main__":
    main()
