from typing import Any, Dict, List

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


CHROMA_DIR = "storage/chroma_referencement"


def get_referencement_vectorstore():
    """
    Load the persisted local Chroma index.
    """

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="referencement",
    )


def retrieve_referencement_context(
    query: str,
    insurers: List[str],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant referencing context from local PDF index.

    The retriever first performs semantic search, then filters by insurer
    when insurer metadata is available.
    """

    vectorstore = get_referencement_vectorstore()

    search_query = query

    if insurers:
        search_query += "\nTarget insurers: " + ", ".join(insurers)

    docs = vectorstore.similarity_search(
        search_query,
        k=max(k * 3, k),
    )

    insurer_set = {x.upper() for x in insurers}

    filtered_docs = []

    for doc in docs:
        doc_insurer = str(doc.metadata.get("insurer", "")).upper()

        if insurer_set and doc_insurer not in insurer_set:
            continue

        filtered_docs.append(doc)

        if len(filtered_docs) >= k:
            break

    if insurer_set and not filtered_docs:
        return [
            {
                "source_type": "referencing_charter",
                "entity": insurer,
                "source_name": None,
                "source_path": None,
                "page": None,
                "content": "",
                "error": "No referencing charter found for this insurer in the local PDF index.",
            }
            for insurer in insurers
        ]

    return [
        {
            "source_type": doc.metadata.get("source_type", "referencing_charter"),
            "entity": doc.metadata.get("insurer"),
            "source_name": doc.metadata.get("source"),
            "source_path": doc.metadata.get("source_path"),
            "page": doc.metadata.get("page"),
            "content": doc.page_content,
        }
        for doc in filtered_docs
    ]