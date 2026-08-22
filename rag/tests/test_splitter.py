import pytest

from rag.dataloader.data import Document
from rag.splitter import SplitConfig, TextSplitter


class PipeTextSplitter(TextSplitter):
    """Minimal concrete splitter used to test TextSplitter's shared behavior."""

    def split_text(self, text: str) -> list[str]:
        return self.merge_splits(text.split("|"), " ")


def test_chunk_overlap_is_smaller_than_chunk_size():
    with pytest.raises(RuntimeError):
        PipeTextSplitter(SplitConfig(chunk_size=3, chunk_overlap=3))


def test_merge_splits_respects_chunk_size_and_overlap():
    splitter = PipeTextSplitter(SplitConfig(chunk_size=7, chunk_overlap=3))

    chunks = splitter.merge_splits(["aaa", "bbb", "ccc"], " ")

    assert chunks == ["aaa bbb", "bbb ccc"]


def test_split_documents_creates_chunk_documents_with_metadata():
    splitter = PipeTextSplitter(SplitConfig(chunk_size=7, chunk_overlap=3))
    loaded_document = Document(
        page_content="aaa|bbb|ccc",
        metadata={"source": "example.pdf"},
    )

    chunks = splitter.split_documents([loaded_document])

    assert [chunk.page_content for chunk in chunks] == ["aaa bbb", "bbb ccc"]
    assert chunks[0].metadata == {
        "chunk": 0,
        "total_chunks": 2,
        "source": "example.pdf",
    }
    assert chunks[1].metadata == {
        "chunk": 1,
        "total_chunks": 2,
        "source": "example.pdf",
    }


def test_merge_splits_uses_length_func_when_removing_overlap():
    word_count = lambda text: len(text.split())
    splitter = PipeTextSplitter(
        SplitConfig(chunk_size=5, chunk_overlap=1, length_func=word_count)
    )

    chunks = splitter.merge_splits(
        ["longword", "secondword", "thirdword", "fourthword", "fifthword"],
        " ",
    )

    assert chunks == [
        "longword secondword thirdword",
        "thirdword fourthword fifthword",
    ]


def test_merge_splits_excludes_empty_chunks():
    splitter = PipeTextSplitter(SplitConfig(chunk_size=7, chunk_overlap=3))

    assert splitter.merge_splits(["", ""], " ") == []
