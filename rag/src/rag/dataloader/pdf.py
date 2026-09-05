from typing import cast

import pymupdf
import requests

from .data import DataLoader, Document


class PDFLoader(DataLoader):
    def __init__(self, source_uri: str, split_pages=False):
        self.pdf_document: pymupdf.Document | None = None
        self.source_uri = source_uri
        self.split_pages = split_pages
        if source_uri.startswith(("http://", "https://")):
            r = requests.get(source_uri)
            if r.status_code != 200:
                raise RuntimeError(
                    f"failed to get pdf from {source_uri}: {r.status_code}"
                )

            self.pdf_document = pymupdf.Document(stream=r.content)

        elif source_uri.startswith("file://"):
            file_path = source_uri.replace("file://", "")
            self.pdf_document = pymupdf.open(file_path)

        else:
            raise RuntimeError(f"unknown source: {source_uri}")

    def _create_document(self, content: str, page_number: int) -> Document:
        metadata = {}
        if self.pdf_document is not None and type(self.pdf_document.metadata) is dict:
            metadata = self.pdf_document.metadata

        return Document(
            page_content=content,
            metadata={
                "source": self.source_uri,
                "pdf": {
                    "total_pages": len(self.pdf_document)
                    if self.pdf_document is not None
                    else 0,
                    **metadata,
                },
                "loc": {
                    "page_number": page_number,
                },
            },
        )

    def load(self) -> list[Document]:
        documents: list[Document] = []
        full_content = ""

        if self.pdf_document is None:
            raise RuntimeError("no documents loaded")

        for page in self.pdf_document:
            text = cast(str, page.get_text("text"))
            if not self.split_pages:
                full_content = (
                    text if full_content == "" else f"{full_content}\n\n{text}"
                )
                continue

            page_number = cast(int, page.number) + 1 if page.number is not None else 0
            documents.append(self._create_document(text, page_number))

        if not self.split_pages:
            documents = [self._create_document(full_content, 1)]

        return documents
