import pytest

from rag.dataloader.data import Document
from rag.splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SplitConfig,
    TextSplitter,
    TokenTextSplitter,
)


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


def test_character_text_splitter_uses_separator_and_overlap():
    splitter = CharacterTextSplitter(
        SplitConfig(chunk_size=7, chunk_overlap=3), separator="|"
    )

    assert splitter.split_text("aaa|bbb|ccc") == ["aaa|bbb", "bbb|ccc"]


def test_recursive_character_text_splitter_preserves_oversized_splits():
    splitter = RecursiveCharacterTextSplitter(
        SplitConfig(chunk_size=5, chunk_overlap=0),
        separators=("|", " ", ""),
    )

    chunks = splitter.split_text("ok|this is oversized|end")

    assert chunks == ["ok", "this", "is", "overs", "ized", "end"]


def test_recursive_character_text_splitter_falls_back_to_characters():
    splitter = RecursiveCharacterTextSplitter(
        SplitConfig(chunk_size=4, chunk_overlap=0)
    )

    assert splitter.split_text("abcdefghij") == ["abcd", "efgh", "ij"]


def test_token_text_splitter_uses_approximate_token_length():
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=2,
        chunk_overlap=0,
    )

    assert splitter.config.length_func("12345") == 2
    assert splitter.split_text("abcdefghij") == ["ab", "cd", "ef", "gh", "ij"]
