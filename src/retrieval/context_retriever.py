from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


VECTOR_ROOT = Path("data_agent_vect/referencement")
CHROMA_DIR = VECTOR_ROOT / "chroma"
COLLECTION_NAME = "referencement_context"


def get_context_vectorstore() -> Chroma:
    """
    Load the persisted local Chroma index.

    The index must be built offline using:
        python3 src/retrieval/build_data_index.py
    """

    if not CHROMA_DIR.exists():
        raise RuntimeError(
            "Referencement vector index not found. "
            "Run: python3 src/retrieval/build_data_index.py"
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def retrieve_context_from_index(
    query: str,
    insurers: Optional[List[str]] = None,
    source_types: Optional[List[str]] = None,
    k: int = 6,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from the local multi-source Chroma index.

    Filtering is applied after semantic search using metadata:
    - insurer
    - source_type
    """

    vectorstore = get_context_vectorstore()

    insurers_upper = {x.upper() for x in insurers or []}
    source_type_set = set(source_types or [])

    search_query = query

    if insurers:
        search_query += "\nInsurers: " + ", ".join(insurers)

    if source_types:
        search_query += "\nSource types: " + ", ".join(source_types)

    docs = vectorstore.similarity_search(
        search_query,
        k=max(k * 5, k),
    )

    filtered_docs = []

    for doc in docs:
        doc_insurer = str(doc.metadata.get("insurer", "")).upper()
        doc_source_type = str(doc.metadata.get("source_type", ""))

        if insurers_upper and doc_insurer not in insurers_upper:
            continue

        if source_type_set and doc_source_type not in source_type_set:
            continue

        filtered_docs.append(doc)

        if len(filtered_docs) >= k:
            break

    if (insurers_upper or source_type_set) and not filtered_docs:
        return [
            {
                "business_domain": "referencement",
                "source_type": source_type,
                "entity": insurer,
                "source_name": None,
                "source_path": None,
                "page": None,
                "content": "",
                "error": "No matching context found in local vector index.",
            }
            for insurer in (insurers or [None])
            for source_type in (source_types or [None])
        ]

    return [
        {
            "business_domain": doc.metadata.get("business_domain"),
            "source_type": doc.metadata.get("source_type"),
            "entity": doc.metadata.get("insurer"),
            "source_name": doc.metadata.get("source_name"),
            "source_path": doc.metadata.get("source_path"),
            "page": doc.metadata.get("page"),
            "content": doc.page_content,
        }
        for doc in filtered_docs
    ]