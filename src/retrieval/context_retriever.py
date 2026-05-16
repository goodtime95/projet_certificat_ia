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

def build_insurer_filter(
    insurer: Optional[str] = None,
) -> Optional[Dict[str, Any]]:

    if not insurer:
        return None

    return {
        "insurer": {
            "$eq": insurer.upper()
        }
    }

def retrieve_context_from_index(
    query: str,
    insurers: Optional[List[str]] = None,
    k_per_insurer: int = 5,
) -> List[Dict[str, Any]]:

    vectorstore = get_context_vectorstore()

    insurers_list = insurers or [None]

    retrieved_items: List[Dict[str, Any]] = []

    for insurer in insurers_list:

        chroma_filter = build_insurer_filter(insurer)

        search_kwargs = {
            "k": k_per_insurer,
        }

        if chroma_filter:
            search_kwargs["filter"] = chroma_filter

        docs = vectorstore.similarity_search(
            query,
            **search_kwargs,
        )

        if not docs:
            retrieved_items.append(
                {
                    "business_domain": "referencement",
                    "entity": insurer,
                    "source_name": None,
                    "source_path": None,
                    "page": None,
                    "content": "",
                    "error": "No matching context found in local vector index.",
                }
            )
            continue

        for doc in docs:
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