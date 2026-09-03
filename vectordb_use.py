from rag.dataloader.data import Document
from rag.embedding import EmbeddingModel, ModelChoice
from rag.vector_db_lighter import VectorDB, SearchResult


def prepare_program_example_documents() -> list[Document]:
    return [
        Document(
            page_content="Python is a high-level programming language known for its simplicity.",
            metadata={
                "id": "doc_1",
                "category": "programming",
                "language": "python",
                "difficulty": "beginner",
            },
        ),
        Document(
            page_content="JavaScript is essential for web development and runs in browsers.",
            metadata={
                "id": "doc_2",
                "category": "programming",
                "language": "javascript",
                "difficulty": "beginner",
            },
        ),
        Document(
            page_content="Machine learning models require training data and computational resources.",
            metadata={
                "id": "doc_3",
                "category": "ai",
                "topic": "machine-learning",
                "difficulty": "intermediate",
            },
        ),
        Document(
            page_content="Neural networks are inspired by biological neurons in the brain.",
            metadata={
                "id": "doc_4",
                "category": "ai",
                "topic": "deep-learning",
                "difficulty": "advanced",
            },
        ),
        Document(
            page_content="React is a popular JavaScript library for building user interfaces.",
            metadata={
                "id": "doc_5",
                "category": "programming",
                "language": "javascript",
                "difficulty": "intermediate",
            },
        ),
        Document(
            page_content="Natural language processing enables computers to understand human language.",
            metadata={
                "id": "doc_6",
                "category": "ai",
                "topic": "nlp",
                "difficulty": "intermediate",
            },
        ),
        Document(
            page_content="Docker containers provide isolated environments for running applications.",
            metadata={
                "id": "doc_7",
                "category": "devops",
                "topic": "containerization",
                "difficulty": "intermediate",
            },
        ),
        Document(
            page_content="SQL databases use structured query language for data management.",
            metadata={
                "id": "doc_8",
                "category": "database",
                "topic": "sql",
                "difficulty": "beginner",
            },
        ),
        Document(
            page_content="Kubernetes orchestrates containerized applications across clusters.",
            metadata={
                "id": "doc_9",
                "category": "devops",
                "topic": "orchestration",
                "difficulty": "advanced",
            },
        ),
        Document(
            page_content="TypeScript adds static typing to JavaScript for better code quality.",
            metadata={
                "id": "doc_10",
                "category": "programming",
                "language": "typescript",
                "difficulty": "intermediate",
            },
        ),
    ]


class ProgramStoreExample:
    def __init__(self) -> None:
        self.embedding_model = EmbeddingModel(ModelChoice.BGE_SMALL_1_5)
        self.vector_store = VectorDB(dim=384, max_elements=10000)
        self.namespace = "memory"

        for doc in prepare_program_example_documents():
            self.vector_store.insert(
                namespace=self.namespace,
                id=doc.metadata.get("id", ""),
                vector=self.embedding_model.embed(doc.page_content),
                metadata={
                    "content": doc.page_content,
                    **doc.metadata,
                },
            )

    def search(self, text: str, top_k: int) -> list[SearchResult]:
        return self.vector_store.search(
            self.namespace, self.embedding_model.embed(text), top_k
        )


def basic_similarity_search():
    store = ProgramStoreExample()
    queries = [
        "How do I learn programming?",
        "Tell me about artificial intelligence",
        "Container deployment tools",
    ]

    for q in queries:
        print(f"Query: {q}")
        results = store.search(q, 3)

        print("Top 3 Results:")
        print("-" * 70)
        for r in results:
            print(f"[Score: {r.get('similarity')}]")
            print(f"\tID: {r.get('id')}")
            print(f"\tContent:{r.get('metadata')['content'][:60]}...")
            print(f"\tCategory:{r.get('metadata')['category']}")


if __name__ == "__main__":
    basic_similarity_search()
