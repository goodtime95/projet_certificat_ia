from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


PDF_DIR = Path("data/referencement")
CHROMA_DIR = Path("storage/chroma_referencement")


def infer_insurer_from_filename(path: Path) -> str:
    """
    Infer insurer name from PDF filename.

    Example:
    axa.pdf -> AXA
    generali.pdf -> GENERALI
    """

    return path.stem.upper()


def load_pdf_documents():
    """
    Load all PDF files from the local referencing directory.
    """

    documents = []

    for pdf_path in PDF_DIR.glob("*.pdf"):
        insurer = infer_insurer_from_filename(pdf_path)

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata["source_type"] = "referencing_charter"
            page.metadata["insurer"] = insurer
            page.metadata["source_path"] = str(pdf_path)

        documents.extend(pages)

    return documents


def build_index():
    """
    Build a local Chroma index from referencing PDF documents.
    """

    documents = load_pdf_documents()

    if not documents:
        raise ValueError(f"No PDF documents found in {PDF_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="referencement",
    )

    print(f"Loaded PDFs: {len(list(PDF_DIR.glob('*.pdf')))}")
    print(f"Indexed chunks: {len(chunks)}")
    print(f"Persisted Chroma index to: {CHROMA_DIR}")

    return vectorstore


if __name__ == "__main__":
    build_index()