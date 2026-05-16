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
    ConversationContext,
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

RETRIEVAL_INTENTS = {
    Intent.REFERENCING_FEASIBILITY,
    Intent.CONSTRAINT_SUMMARY,
    Intent.POLICY_CONFIRMATION,
    Intent.MEMORY_OR_HISTORY,
}

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

            history = state.get("conversation_history", [])
            conversation_context = state.get("conversation_context")

            context_payload = (
                conversation_context.model_dump()
                if conversation_context
                else None
            )

            messages = [
                {
                    "role": "system",
                    "content": INTERPRETATION_SYSTEM_PROMPT,
                }
            ]

            messages.extend(history[-6:])

            messages.append(
                {
                    "role": "user",
                    "content": f"""
        REQUÊTE UTILISATEUR :
        {query}

        CONTEXTE MÉTIER ACTIF :
        {context_payload}

        Instruction :
        Si la requête utilisateur est elliptique alors utilise le CONTEXTE MÉTIER ACTIF pour reconstruire la demande complète.
        Ne classe pas la demande en UNCLEAR si le contexte métier actif permet de comprendre :
        - le produit,
        - l’assureur,
        - ou la question métier.
        """
                }
            )
            # print("\n" + "=" * 80)
            # print("DEBUG - INTERPRETER INPUT MESSAGES")
            # print("=" * 80)

            # for i, message in enumerate(messages):
            #     print(f"\n--- MESSAGE {i} ---")
            #     print(f"role: {message.get('role')}")
            #     print(message.get("content"))

            result = structured_llm.invoke(messages)

            # print("\n" + "=" * 80)
            # print("DEBUG - INTERPRETER OUTPUT")
            # print("=" * 80)
            # print(result.model_dump_json(indent=2))
            # print("=" * 80 + "\n")

            previous_context = state.get("conversation_context")

            updated_context = update_conversation_context(
                previous_context=previous_context,
                interpreted=result,
            )

            return {
                "interpreted_request": result,
                "conversation_context": updated_context,
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

    if interpreted.intent in RETRIEVAL_INTENTS:
        return "retrieve_context"

    if interpreted.required_sources:
        return "retrieve_context"

    return "generate_answer"

def retrieve_context(state: AgentV2State) -> dict:
    interpreted = state.get("interpreted_request")
    user_query = state.get("user_query", "")
    retrieval_query = (
        interpreted.standalone_query
        if interpreted and interpreted.standalone_query
        else user_query
    )

    if interpreted is None:
        return {
            "retrieved_context": [],
            "missing_context_sources": [],
            "error": "No interpreted request available for retrieval.",
        }

    insurers = interpreted.insurers or []

    retrieved_context = retrieve_context_from_index(
        query=retrieval_query,
        insurers=insurers,
        k_per_insurer=5,
    )

    sources_found = sorted(
        {
            chunk.get("source_type")
            for chunk in retrieved_context
            if chunk.get("source_type")
        }
    )

    entities_found = sorted(
        {
            chunk.get("entity")
            for chunk in retrieved_context
            if chunk.get("entity")
        }
    )

    retrieval_status = {
        "chunks_found": len(retrieved_context),
        "insurers_requested": insurers,
        "entities_found": entities_found,
        "sources_found": sources_found,
    }

    return {
        "retrieved_context": retrieved_context,
        "retrieval_status": retrieval_status,
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
                            f"{retrieval_status if retrieval_status else {'status': 'not_run'}}\n\n"
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


def update_conversation_context(
    previous_context: ConversationContext | None,
    interpreted: InterpretedRequest,
) -> ConversationContext:
    previous_context = previous_context or ConversationContext()

    active_insurers = (
        interpreted.insurers
        if interpreted.insurers
        else previous_context.active_insurers
    )

    active_products = (
        interpreted.products
        if interpreted.products
        else previous_context.active_products
    )

    active_topic = interpreted.intent.value if interpreted.intent else previous_context.active_topic

    return ConversationContext(
        active_insurers=active_insurers,
        active_products=active_products,
        active_topic=active_topic,
        last_intent=interpreted.intent,
    )
