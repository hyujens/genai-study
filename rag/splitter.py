from abc import ABC, abstractmethod
from collections.abc import Callable, Sized
from dataclasses import dataclass

from .dataloader.data import Document

ChunkDocument = Document


@dataclass
class SplitConfig:
    # Maximum size of each chunk
    chunk_size: int = 1000
    # How much content to repeat between chunks
    chunk_overlap: int = 200
    # How to measure text length
    length_func: Callable[[Sized], int] = lambda t: len(t)
    # Whether to include separators in chunks
    keep_separator: bool = False


class TextSplitter(ABC):
    def __init__(self, config: SplitConfig) -> None:
        # chunk overlap must not be larger than or equal to chunk size.
        # Otherwise, it will cause infinitely splitting text.
        if config.chunk_overlap >= config.chunk_size:
            raise RuntimeError("chunk overlap must be less than chunk size")

        self.config = config

    @abstractmethod
    def split_text(self, text: str) -> list[str]: ...

    def create_documents(self, loaded_documents: list[Document]) -> list[ChunkDocument]:
        """Converts raw text segments into Document objects with metadata."""
        documents: list[ChunkDocument] = []
        for doc in loaded_documents:
            chunks = self.split_text(doc.page_content)

            for idx, chunk in enumerate(chunks):
                documents.append(
                    ChunkDocument(
                        page_content=chunk,
                        metadata={
                            "chunk": idx,
                            "total_chunks": len(chunks),
                            **doc.metadata,
                        },
                    )
                )

        return documents

    def split_documents(self, documents: list[Document]) -> list[ChunkDocument]:
        """Splits a list of Document objects into chunked Documents."""
        return self.create_documents(documents)

    def join_splits(self, splits: list[str], separator: str) -> str:
        """Joins text splits with a separator and trims whitespace."""
        return separator.join(splits).strip()

    def merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merges text splits into chunks with overlap."""
        chunks: list[str] = []
        candidate_splits_for_chunk: list[str] = []
        accumulated_chunk_len = 0

        for split in splits:
            split_len = self.config.length_func(split)
            separator_len_for_add = len(separator) if candidate_splits_for_chunk else 0

            next_accumulated_chunk_len = (
                accumulated_chunk_len + split_len + separator_len_for_add
            )
            if next_accumulated_chunk_len > self.config.chunk_size:
                if candidate_splits_for_chunk:
                    chunks.append(
                        self.join_splits(candidate_splits_for_chunk, separator)
                    )

                while (
                    accumulated_chunk_len > self.config.chunk_overlap
                    and candidate_splits_for_chunk
                ):
                    throw_out_split_candidate = candidate_splits_for_chunk.pop(0)
                    accumulated_chunk_len -= self.config.length_func(
                        throw_out_split_candidate
                    ) + len(separator)

            # calculate again since some candidates have been moved out
            separator_len_for_add = len(separator) if candidate_splits_for_chunk else 0
            accumulated_chunk_len += split_len + separator_len_for_add

            candidate_splits_for_chunk.append(split)

        if candidate_splits_for_chunk:
            chunks.append(self.join_splits(candidate_splits_for_chunk, separator))

        return [chunk for chunk in chunks if chunk]
