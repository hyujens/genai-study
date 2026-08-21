import re
from dataclasses import dataclass, field
from typing import Any, Protocol


class DataLoader(Protocol):
    def load(self) -> list[Document]: ...


@dataclass
class Document:
    # the cleaned text content extracted from a document
    page_content: str
    # document information
    metadata: dict[str, Any] = field(default_factory=dict)
    # optional identifier for the document
    id: str | None = None


def clean_document(document: Document) -> Document:
    content = document.page_content

    # Remove page markers like "-- 1 of 22 --" or "— 5 of 12 —"
    content = re.sub(
        r"[-–—]\s*\d+\s*of\s*\d+\s*[-–—]", "", content, flags=re.IGNORECASE
    )

    # Normalize multiple spaces and tabs
    content = re.sub(r"[ \t]+", " ", content, flags=re.IGNORECASE)

    # Preserve paragraph breaks but remove unnecessary empty lines
    content = re.sub(r"\n{3,}", "\n\n", content, flags=re.IGNORECASE)
    document.page_content = content.strip()

    return document
