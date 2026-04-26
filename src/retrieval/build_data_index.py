import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


RAW_DATA_ROOT = Path("data_agent/referencement")
VECTOR_ROOT = Path("data_agent_vect/referencement")
CHROMA_DIR = VECTOR_ROOT / "chroma"
MANIFEST_PATH = VECTOR_ROOT / "manifest.json"

COLLECTION_NAME = "referencement_context"

SOURCE_TYPE_BY_FOLDER = {
    "chartes": "referencing_charter",
    "emails": "email_history",
    "notes": "internal_note",
}


def load_pdf(path: Path, metadata: Dict) -> List[Document]:
    """
    Load a PDF file and attach business metadata to each page.
    """

    loader = PyPDFLoader(str(path))
    docs = loader.load()

    for doc in docs:
        doc.metadata.update(metadata)
        doc.metadata["page"] = doc.metadata.get("page")

    return docs


def load_text(path: Path, metadata: Dict) -> List[Document]:
    """
    Load a Markdown or text file and attach business metadata.
    """

    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()

    for doc in docs:
        doc.metadata.update(metadata)
        doc.metadata["page"] = None

    return docs


def load_raw_documents() -> List[Document]:
    """
    Load supported raw documents from data_agent/referencement.

    Expected structure:
        data_agent/referencement/{insurer}/{chartes|emails|notes}/file.pdf|md|txt
    """

    documents: List[Document] = []

    if not RAW_DATA_ROOT.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DATA_ROOT}")

    for insurer_dir in RAW_DATA_ROOT.iterdir():
        if not insurer_dir.is_dir():
            continue

        insurer = insurer_dir.name.upper()

        for folder_name, source_type in SOURCE_TYPE_BY_FOLDER.items():
            source_dir = insurer_dir / folder_name

            if not source_dir.exists():
                continue

            for file_path in source_dir.rglob("*"):
                if not file_path.is_file():
                    continue

                if file_path.suffix.lower() not in {".pdf", ".md", ".txt"}:
                    continue

                metadata = {
                    "business_domain": "referencement",
                    "insurer": insurer,
                    "source_type": source_type,
                    "source_folder": folder_name,
                    "source_name": file_path.name,
                    "source_path": str(file_path),
                }

                if file_path.suffix.lower() == ".pdf":
                    documents.extend(load_pdf(file_path, metadata))

                else:
                    documents.extend(load_text(file_path, metadata))

    return documents


def write_manifest(documents: List[Document], chunks_count: int) -> None:
    """
    Write a manifest file describing what was indexed.
    """

    indexed_files = {}

    for doc in documents:
        source_path = doc.metadata.get("source_path")
        if not source_path:
            continue

        indexed_files[source_path] = {
            "business_domain": doc.metadata.get("business_domain"),
            "insurer": doc.metadata.get("insurer"),
            "source_type": doc.metadata.get("source_type"),
            "source_folder": doc.metadata.get("source_folder"),
            "source_name": doc.metadata.get("source_name"),
        }

    manifest = {
        "built_at": datetime.now().isoformat(),
        "raw_data_root": str(RAW_DATA_ROOT),
        "vector_root": str(VECTOR_ROOT),
        "collection_name": COLLECTION_NAME,
        "documents_loaded": len(documents),
        "chunks_indexed": chunks_count,
        "indexed_files": list(indexed_files.values()),
    }

    VECTOR_ROOT.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def build_data_index() -> None:
    """
    Build the local Chroma index from immutable raw data.

    This function is an offline ingestion step.
    The runtime agent should only read the generated vector index.
    """

    documents = load_raw_documents()
    unique_files = {
    doc.metadata.get("source_path")
    for doc in documents
    if doc.metadata.get("source_path")
}

    if not documents:
        raise ValueError(f"No supported documents found in {RAW_DATA_ROOT}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    VECTOR_ROOT.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    write_manifest(documents=documents, chunks_count=len(chunks))

    print(f"Loaded files: {len(unique_files)}")
    print(f"Loaded pages/text units: {len(documents)}")
    print(f"Indexed chunks: {len(chunks)}")
    print(f"Persisted Chroma index to: {CHROMA_DIR}")
    print(f"Wrote manifest to: {MANIFEST_PATH}")


if __name__ == "__main__":
    build_data_index()