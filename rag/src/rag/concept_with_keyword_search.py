# assume we save knowledge somewhere
knowledge = [
    "Underwhelming Spatula is a kitchen tool that redefines expectations by fusing whimsy with functionality.",
    "Lisa Melton wrote Dubious Parenting Tips.",
    "The Almost-Perfect Investment Guide is 210 pages long.",
    "Quantum computing uses qubits instead of classical bits.",
    "The capital of France is Paris.",
]


def naive_keyword_search(query: str, documents: list[str], top_k=2) -> list[str]:
    # remove common words since they are not what we want during retrieval
    stop_words = {"a", "an", "the", "is", "are", "what", "who"}
    words_in_query = [w for w in query.lower().split() if w not in stop_words]

    scored: list[dict] = []
    for doc in documents:
        words_in_doc = doc.lower().split()
        scored.append(
            {
                "doc": doc,
                "score": sum(1 for word in words_in_query if word in words_in_doc),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)

    return [item["doc"] for item in scored[:top_k] if item["score"] > 0]
