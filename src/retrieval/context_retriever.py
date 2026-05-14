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
    k_per_source: int = 3,
) -> List[Dict[str, Any]]:
    
    """
    Retrieve relevant chunks from the local multi-source Chroma index.

    Retrieval is balanced by:
    - insurer
    - source type

    This prevents one insurer with highly similar chunks from dominating
    the retrieved context in multi-insurer questions.
    """

    vectorstore = get_context_vectorstore()

    insurers_list = insurers or [None]
    source_types_list = source_types or [None]

    retrieved_items: List[Dict[str, Any]] = []

    for insurer in insurers_list:
        for source_type in source_types_list:
            search_query = query

            if insurer:
                search_query += f"\nTarget insurer: {insurer}"

            if source_type:
                search_query += f"\nSource type: {source_type}"

            docs = vectorstore.similarity_search(
                search_query,
                k=max(k_per_source * 5, k_per_source),
            )

            filtered_docs = []

            for doc in docs:
                doc_insurer = str(doc.metadata.get("insurer", "")).upper()
                doc_source_type = str(doc.metadata.get("source_type", ""))

                if insurer and doc_insurer != insurer.upper():
                    continue

                if source_type and doc_source_type != source_type:
                    continue

                filtered_docs.append(doc)

                if len(filtered_docs) >= k_per_source:
                    break

            if not filtered_docs:
                retrieved_items.append(
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
                )
                continue

            for doc in filtered_docs:
                retrieved_items.append(
                    {
                        "business_domain": doc.metadata.get("business_domain"),
                        "source_type": doc.metadata.get("source_type"),
                        "entity": doc.metadata.get("insurer"),
                        "source_name": doc.metadata.get("source_name"),
                        "source_path": doc.metadata.get("source_path"),
                        "page": doc.metadata.get("page"),
                        "content": doc.page_content,
                    }
                )

    return retrieved_items