from pathlib import Path

from rag.dataloader import data, pdf


def main():
    pdf_path = Path.home() / "Downloads" / "discourse_on_the_Method.pdf"
    loader = pdf.PDFLoader(pdf_path.as_uri(), split_pages=False)
    documents = [data.clean_document(doc) for doc in loader.load()]

    print(documents[0].page_content[:300])


if __name__ == "__main__":
    main()
