import json
from pathlib import Path

from rag.dataloader.data import Document
from rag.embedding import EmbeddingModel


class FakeLlama:
    def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


def create_embedding_model() -> EmbeddingModel:
    """Create the wrapper without loading a real GGUF model."""
    model = EmbeddingModel.__new__(EmbeddingModel)
    model.model = "test-embedding-model.gguf"
    model.llama = FakeLlama()
    return model


def test_generate_builds_embedding_records_and_reports_progress():
    model = create_embedding_model()
    documents = [
        Document(page_content="first", metadata={"id": "first-id"}),
        Document(page_content="second", metadata={"topic": "example"}),
    ]
    progress: list[tuple[int, int]] = []

    embeddings = model.generate(
        documents,
        on_progress=lambda processed, total: progress.append((processed, total)),
    )

    assert [item["id"] for item in embeddings] == ["first-id", "doc_1"]
    assert [item["embedding"] for item in embeddings] == [[5.0, 1.0], [6.0, 1.0]]
    assert embeddings[1]["metadata"] == {"topic": "example"}
    assert progress == [(1, 2), (2, 2)]


def test_generate_uses_default_id_when_metadata_id_is_falsy():
    model = create_embedding_model()
    document = Document(page_content="content", metadata={"id": None})

    embeddings = model.generate([document])

    assert embeddings[0]["id"] == "doc_0"


def test_save_as_json_can_be_loaded_back(tmp_path: Path):
    model = create_embedding_model()
    embeddings = model.generate([Document(page_content="content")])
    output_file = tmp_path / "embeddings.json"

    result = model.save_as_json(embeddings, str(output_file))
    loaded = model.load_from_file(str(output_file))
    saved_data = json.loads(output_file.read_text(encoding="utf-8"))

    assert loaded == embeddings
    assert result["filepath"] == str(output_file)
    assert result["size"] == output_file.stat().st_size
    assert saved_data["version"] == "1.0"
    assert saved_data["model"] == "test-embedding-model.gguf"
    assert saved_data["dimensions"] == 2
    assert saved_data["count"] == 1
    assert isinstance(saved_data["created"], str)


def test_build_lookup_from_missing_file_returns_empty_lookup(tmp_path: Path):
    model = create_embedding_model()

    lookup = model.build_lookup_from_file(str(tmp_path / "missing.json"))

    assert lookup == {}


def test_build_lookup_indexes_embeddings_by_id(tmp_path: Path):
    model = create_embedding_model()
    output_file = tmp_path / "embeddings.json"
    output_file.write_text(
        '{"embeddings": [{"id": "doc-1", "embedding": [1.0, 2.0]}]}',
        encoding="utf-8",
    )

    lookup = model.build_lookup_from_file(str(output_file))

    assert lookup == {"doc-1": {"id": "doc-1", "embedding": [1.0, 2.0]}}


def test_update_only_generates_embeddings_for_new_documents():
    model = create_embedding_model()
    existing = {
        "id": "existing-id",
        "content": "existing",
        "metadata": {"id": "existing-id"},
        "embedding": [8.0, 1.0],
        "timestamp": 1.0,
    }
    model.build_lookup_from_file = lambda _: {"existing-id": existing}
    documents = [
        Document(page_content="existing", metadata={"id": "existing-id"}),
        Document(page_content="new", metadata={"id": "new-id"}),
    ]

    embeddings = model.update(documents, "unused.json")

    assert [item["id"] for item in embeddings] == ["existing-id", "new-id"]
