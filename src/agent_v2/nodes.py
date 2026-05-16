from langchain_openai import ChatOpenAI

from src.agent_v2.prompts import (
    INTERPRETATION_SYSTEM_PROMPT,
    FAST_RESPONSE_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
)

from src.agent_v2.schemas import (
    InterpretedRequest,
    AgentAnswer,
    Intent,
    ResponseMode,
    SourceReference,
)

from src.agent_v2.state import AgentV2State

from src.retrieval.context_retriever import retrieve_context_from_index

from pathlib import Path
from typing import Any, Dict, List


MEMORY_FILES = {
    "persistent_memory": Path("data_agent/memory/persistent_memory.md"),
}

REFERENCEMENT_DIR = Path("data/referencement")

EMAIL_HISTORY_DIR = Path("data/email_history_mock")

def _read_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()

def make_interpret_user_request_node(model_name: str = "gpt-4.1"):

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    structured_llm = llm.with_structured_output(InterpretedRequest)

    def interpret_user_request(state: AgentV2State):

        query = state["user_query"]

        try:

            result = structured_llm.invoke(
                [
                    {"role": "system", "content": INTERPRETATION_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ]
            )

            return {
                "interpreted_request": result,
                "model_used": model_name,
            }

        except Exception as e:

            return {
                "error": str(e)
            }

    return interpret_user_request

def route_after_interpretation(state: AgentV2State) -> str:
    interpreted = state.get("interpreted_request")

    if interpreted is None:
        return "generate_fast_response"

    if interpreted.intent in {
        Intent.OUT_OF_SCOPE,
        Intent.UNCLEAR,
        Intent.PRODUCT_ADVICE,
    }:
        return "generate_fast_response"

    return "load_memory_context"

def make_generate_fast_response_node(model_name: str = "gpt-4.1-mini"):
    llm = ChatOpenAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(AgentAnswer)
    def generate_fast_response(state: AgentV2State) -> dict:
        interpreted = state.get("interpreted_request")
        user_query = state.get("user_query", "")
        result = structured_llm.invoke(
            [
                ("system", FAST_RESPONSE_SYSTEM_PROMPT),
                (
                    "user",
                    f"""
                Requête utilisateur :
                {user_query}

                Interprétation structurée :
                {interpreted.model_dump() if interpreted else None}

                Produis une réponse courte adaptée à cette interprétation.
                """
                ),
            ]
        )
        return {
            "answer": result,
        }
    return generate_fast_response

def load_memory_context(state: AgentV2State) -> AgentV2State:
    sections = []

    for name, path in MEMORY_FILES.items():
        content = _read_file(path)
        if content:
            sections.append(f"# {name}\n{content}")

    return {
        "memory_context": "\n\n".join(sections) or "No memory context found."
    }

def route_after_memory(state: AgentV2State) -> str:
    interpreted = state.get("interpreted_request")

    if interpreted is None:
        return "generate_fast_response"

    if not interpreted.required_sources:
        return "generate_answer"

    return "retrieve_context"

def retrieve_context(state: AgentV2State) -> dict:
    interpreted = state.get("interpreted_request")
    user_query = state.get("user_query", "")

    if interpreted is None:
        return {
            "retrieved_context": [],
            "missing_context_sources": [],
            "error": "No interpreted request available for retrieval.",
        }

    insurers = interpreted.insurers or []

    retrieved_context = retrieve_context_from_index(
        query=user_query,
        insurers=insurers,
        k_per_insurer=5,
    )

    return {
        "retrieved_context": retrieved_context,
    }

def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "Aucun contexte documentaire récupéré."

    formatted_chunks = []

    for i, chunk in enumerate(chunks, start=1):
        source_id = f"S{i}"

        chunk["source_id"] = source_id

        page = chunk.get("page")
        page_value = page if page is not None else "N/A"

        formatted_chunks.append(
            f"""
[SOURCE_ID]
{source_id}

[ASSUREUR]
{chunk.get("entity") or "UNKNOWN"}

[TYPE_SOURCE]
{chunk.get("source_type") or "unknown_source"}

[SOURCE]
{chunk.get("source_name") or "unknown_file"}

[PAGE]
{page_value}

[CONTENU]
{chunk.get("content") or ""}
"""
        )

    return "\n\n".join(formatted_chunks)

def extract_sources_by_ids(
    retrieved_context: list[dict],
    source_ids: list[str],
) -> list[SourceReference]:

    wanted_ids = set(source_ids or [])
    sources: list[SourceReference] = []
    seen = set()

    for chunk in retrieved_context:
        source_id = chunk.get("source_id")

        if source_id not in wanted_ids:
            continue

        source_type = chunk.get("source_type")
        entity = chunk.get("entity")
        source_name = chunk.get("source_name")
        page = chunk.get("page")

        if not source_type or not source_name:
            continue

        key = (source_type, entity, source_name, page)

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            SourceReference(
                source_type=source_type,
                entity=entity,
                source_name=source_name,
                page=page,
            )
        )

    return sources

def make_generate_answer_node(model_name: str = "gpt-4.1"):

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    structured_llm = llm.with_structured_output(AgentAnswer)

    def generate_answer(state: AgentV2State) -> dict:
        interpreted = state.get("interpreted_request")
        user_query = state.get("user_query", "")
        memory_context = state.get("memory_context", "")
        retrieved_context = state.get("retrieved_context", [])
        retrieval_status = state.get("retrieval_status")

        if interpreted is None:
            fallback = AgentAnswer(
                mode=ResponseMode.CLARIFY,
                answer=(
                    "Je n’ai pas réussi à interpréter la demande. "
                    "Peux-tu préciser le produit, l’assureur cible et la question de référencement ?"
                ),
                missing_information=[
                    "produit",
                    "assureur cible",
                    "question de référencement",
                ],
                sources_used=[],
                confidence="low",
            )

            return {
                "answer": fallback,
            }

        formatted_context = format_context(retrieved_context)

        try:
            result = structured_llm.invoke(
                [
                    {
                        "role": "system",
                        "content": ANSWER_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": (
                            "QUESTION UTILISATEUR :\n"
                            f"{user_query}\n\n"
                            "INTERPRÉTATION STRUCTURÉE :\n"
                            f"{interpreted.model_dump_json(indent=2, exclude_none=True)}\n\n"
                            "MÉMOIRE / CONTEXTE UTILISATEUR :\n"
                            f"{memory_context}\n\n"
                            "STATUT DU RETRIEVAL :\n"
                            f"{retrieval_status or 'unknown'}\n\n"
                            "CONTEXTE DOCUMENTAIRE RÉCUPÉRÉ :\n"
                            f"{formatted_context}"
                        ),
                    },
                ]
            )
            result.sources_used = extract_sources_by_ids(retrieved_context=retrieved_context,
                                                         source_ids=result.source_ids,)

            return {
                "answer": result,
            }

        except Exception as e:
            fallback = AgentAnswer(
                mode=ResponseMode.CLARIFY,
                answer=(
                    "Une erreur technique est survenue pendant la génération de la réponse. "
                    "Je ne peux pas conclure proprement sur la base du contexte récupéré."
                ),
                missing_information=[],
                sources_used=[],
                confidence="low",
            )

            return {
                "answer": fallback,
                "error": str(e),
            }

    return generate_answer

# def summarize_retrieved_context(state: AgentV2State) -> AgentV2State:
#     """
#     Build a compact summary of retrieved context before answer generation.

#     This is not a rule engine.
#     It helps the final LLM distinguish:
#     - issuer rules
#     - underlying rules
#     - wrapper rules
#     - ESG constraints
#     - validation requirements
#     - missing or weak sources

#     It uses normalized retrieval output when available:
#     - valid_retrieved_context
#     - missing_context_sources
#     - weak_context_sources
#     """

#     retrieved_context = state.get(
#         "valid_retrieved_context",
#         state.get("retrieved_context", []),
#     )

#     missing_sources = state.get("missing_context_sources", [])
#     weak_sources = state.get("weak_context_sources", [])

#     lines = []

#     if retrieved_context:
#         lines.append("Retrieved usable context:")

#     for item in retrieved_context:
#         entity = item.get("entity") or "UNKNOWN_ENTITY"
#         source_type = item.get("source_type") or "UNKNOWN_SOURCE_TYPE"
#         source_name = item.get("source_name") or "UNKNOWN_SOURCE"
#         page = item.get("page")
#         content = (item.get("content") or "").strip()

#         if not content:
#             continue

#         header = f"{entity} | {source_type} | {source_name}"
#         if page is not None:
#             header += f" | page {page}"

#         content_lower = content.lower()
#         detected_topics = []

#         topic_rules = {
#             "issuer rules": [
#                 "émetteur",
#                 "emetteur",
#                 "issuer",
#                 "contrepartie",
#                 "counterparty",
#                 "rating",
#                 "notation",
#             ],
#             "underlying rules": [
#                 "sous-jacent",
#                 "sous-jacents",
#                 "underlying",
#                 "underlyings",
#                 "indice",
#                 "indices",
#                 "panier",
#                 "basket",
#                 "titre",
#                 "actions",
#             ],
#             "wrapper rules": [
#                 "uc",
#                 "unit-linked",
#                 "unité de compte",
#                 "unites de compte",
#                 "fonds euro",
#                 "assurance vie",
#                 "capitalisation",
#             ],
#             "ESG constraints": [
#                 "esg",
#                 "exclusion",
#                 "exclusions",
#                 "controverse",
#                 "controverses",
#             ],
#             "validation requirement": [
#                 "validation",
#                 "pré-validation",
#                 "pre-validation",
#                 "agrément",
#                 "agrement",
#                 "accord préalable",
#                 "cas par cas",
#                 "soumis à validation",
#             ],
#             "maturity constraints": [
#                 "maturité",
#                 "maturite",
#                 "tenor",
#                 "durée",
#                 "duree",
#                 "ans",
#                 "années",
#                 "annees",
#             ],
#             "refusal / exclusion rules": [
#                 "refusé",
#                 "refuse",
#                 "refus",
#                 "non-référençable",
#                 "non referencable",
#                 "exclu",
#                 "exclus",
#                 "interdit",
#             ],
#         }

#         for topic, keywords in topic_rules.items():
#             if any(keyword in content_lower for keyword in keywords):
#                 detected_topics.append(topic)

#         if not detected_topics:
#             detected_topics.append("general context")

#         excerpt = " ".join(content.split())
#         if len(excerpt) > 700:
#             excerpt = excerpt[:700] + "..."

#         lines.append(
#             f"- {header}: topics={detected_topics}. Excerpt: {excerpt}"
#         )

#     if missing_sources:
#         lines.append("")
#         lines.append("Missing context sources:")
#         for source in missing_sources:
#             entity = source.get("entity") or "UNKNOWN_ENTITY"
#             source_type = source.get("source_type") or "UNKNOWN_SOURCE_TYPE"
#             message = source.get("message") or "No relevant context retrieved."
#             lines.append(
#                 f"- {entity} | {source_type}: {message}"
#             )

#     if weak_sources:
#         lines.append("")
#         lines.append("Weak context sources:")
#         for source in weak_sources:
#             entity = source.get("entity") or "UNKNOWN_ENTITY"
#             source_type = source.get("source_type") or "UNKNOWN_SOURCE_TYPE"
#             source_name = source.get("source_name") or "UNKNOWN_SOURCE"
#             page = source.get("page")
#             reason = source.get("reason") or "Weak or administrative context."

#             header = f"{entity} | {source_type} | {source_name}"
#             if page is not None:
#                 header += f" | page {page}"

#             lines.append(
#                 f"- {header}: {reason}"
#             )

#     if not lines:
#         return {
#             "context_summary": "No relevant context was retrieved."
#         }

#     return {
#         "context_summary": "\n".join(lines)
#     }


# def make_judge_answer_node(model_name: str = "gpt-4.1"):
#     """
#     Create a final answer judge node.
#     The judge reviews the draft answer and either keeps it or corrects it.
#     """

#     llm = ChatOpenAI(
#         model=model_name,
#         temperature=0,
#     )

#     structured_llm = llm.with_structured_output(AgentAnswer)

#     def judge_answer(state: AgentV2State) -> AgentV2State:
#         answer_draft = state.get("answer_draft")
#         interpreted = state.get("interpreted_request")

#         if answer_draft is None:
#             return {
#                 "error": "No answer_draft available for judge.",
#             }

#         try:
#             result = structured_llm.invoke(
#                 [
#                     {
#                         "role": "system",
#                         "content": JUDGE_SYSTEM_PROMPT,
#                     },
#                     {
#                         "role": "user",
#                         "content": (
#                             "INTERPRETED REQUEST:\n"
#                             f"{interpreted.model_dump_json(indent=2) if interpreted else None}\n\n"
#                             "MEMORY CONTEXT:\n"
#                             f"{state.get('memory_context', '')}\n\n"
#                             "CONTEXT SUMMARY:\n"
#                             f"{state.get('context_summary', '')}\n\n"
#                             "RETRIEVED CONTEXT:\n"
#                             f"{state.get('retrieved_context', [])}\n\n"
#                             "DRAFT ANSWER:\n"
#                             f"{answer_draft.model_dump_json(indent=2)}"
#                         ),
#                     },
#                 ]
#             )

#             return {
#                 "final_answer": result
#             }

#         except Exception as exc:
#             return {
#                 "final_answer": answer_draft,
#                 "error": f"Judge failed, using draft answer: {exc}",
#             }

#     return judge_answer



# def normalize_retrieved_context(state: AgentV2State) -> AgentV2State:
    retrieved_context = state.get("retrieved_context", [])

    valid_context = []
    missing_sources = []
    weak_sources = []

    for item in retrieved_context:
        if item.get("error"):
            missing_sources.append({
                "entity": item.get("entity"),
                "source_type": item.get("source_type"),
                "message": item.get("error"),
            })
            continue

        content = (item.get("content") or "").strip()

        if not content:
            missing_sources.append({
                "entity": item.get("entity"),
                "source_type": item.get("source_type"),
                "message": "Empty retrieved content.",
            })
            continue

        # Source faible : ex. page de contacts, administratif, pas de règles métier
        content_lower = content.lower()
        has_rule_signal = any(
            keyword in content_lower
            for keyword in [
                "émetteur",
                "issuer",
                "sous-jacent",
                "underlying",
                "validation",
                "accepté",
                "refusé",
                "exclusion",
                "esg",
                "uc",
                "fonds euro",
                "maturité",
                "notation",
                "rating",
            ]
        )

        if not has_rule_signal:
            weak_sources.append({
                "entity": item.get("entity"),
                "source_type": item.get("source_type"),
                "source_name": item.get("source_name"),
                "page": item.get("page"),
                "reason": "No clear business rule signal found.",
            })

        valid_context.append(item)

    return {
        "valid_retrieved_context": valid_context,
        "missing_context_sources": missing_sources,
        "weak_context_sources": weak_sources,
    }