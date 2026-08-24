import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from llama_cpp import Llama

from .dataloader.data import Document

BGE_SMALL_1_5 = "bge-small-en-v1.5-q8_0.gguf"


class EmbeddingModel:
    def __init__(self, model: str) -> None:
        self.model = model
        self.llama = Llama(
            model_path=(Path.home() / "workspace/models/embeddings" / model).as_posix(),
            embedding=True,
            verbose=False,
            n_ctx=2048,
            n_gpu_layers=-1,
        )

    def embed(self, text: str) -> list:
        return list(self.llama.embed(text))

    def generate(
        self,
        documents: list[Document],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        embeddings: list[dict] = []

        for processed, document in enumerate(documents):
            embeddings.append(
                {
                    "id": document.metadata.get("id") or f"doc_{processed}",
                    "content": document.page_content,
                    "metadata": document.metadata,
                    "embedding": self.llama.embed(document.page_content),
                    "timestamp": time.time(),
                }
            )

            processed += 1
            if on_progress:
                on_progress(processed, len(documents))

        return embeddings

    def save_as_json(self, embeddings: list[dict], filename: str) -> dict:
        with open(filename, "w") as f:
            f.write(
                json.dumps(
                    {
                        "version": "1.0",
                        "model": self.model,
                        "dimensions": len(embeddings[0]["embedding"])
                        if embeddings and embeddings[0].get("embedding")
                        else 384,
                        "count": len(embeddings),
                        "created": datetime.now(UTC).isoformat(),
                        "embeddings": embeddings,
                    }
                )
            )

        return {
            "filepath": filename,
            "size": Path(filename).stat().st_size,
        }

    def load_from_file(self, filename: str) -> list[dict]:
        data = {}
        with open(filename, "r") as f:
            data = json.load(f)

        return data.get("embeddings", [])

    def build_lookup_from_file(self, filename: str) -> dict[str, dict]:
        lookup = {}
        try:
            embeddings = self.load_from_file(filename)
        except FileNotFoundError:
            return lookup

        for embed in embeddings:
            lookup[embed["id"]] = embed

        return lookup

    def update(self, new_documents: list[Document], existed_file: str) -> list[dict]:
        lookup = self.build_lookup_from_file(existed_file)

        # filter out documents that already have embedding
        embedding_candidates = [
            document
            for document in new_documents
            if (document.metadata.get("id") or document.page_content[:50]) not in lookup
        ]

        embeddings = list(lookup.values())
        if not embedding_candidates:
            return embeddings

        new_embeddings = self.generate(embedding_candidates)
        embeddings.extend(new_embeddings)

        return embeddings
