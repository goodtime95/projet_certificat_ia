from langchain_openai import ChatOpenAI

from src.agent_v2.prompts import (
    INTERPRETATION_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
)

from src.agent_v2.schemas import (
    InterpretedRequest,
    AgentAnswer,
    Intent,
    ResponseMode,
    SourceNeed,
    EvidenceAnalysis,
    EvidenceStatus,
    RuleEvidence,
)

from src.agent_v2.state import AgentV2State

from src.retrieval.context_retriever import retrieve_context_from_index

from pathlib import Path
from typing import Any, Dict, List

MEMORY_FILES = {
    "persistent_memory": Path("data_agent/memory/persistent_memory.md"),
    "business_reference": Path("data_agent/memory/business_reference.md"),
}


REFERENCEMENT_DIR = Path("data/referencement")

INSURER_FILE_MAP = {
    "axa": "axa.md",
    "aep": "aep.md",
    "generali": "generali.md",
}

EMAIL_HISTORY_DIR = Path("data/email_history_mock")

EMAIL_FILE_MAP = {
    "axa": "axa.md",
    "aep": "aep.md",
    "generali": "generali.md",
}


def make_interpret_user_request_node(model_name: str):

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

def _read_memory_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def load_memory_context(state: AgentV2State) -> AgentV2State:
    sections = []

    for name, path in MEMORY_FILES.items():
        content = _read_memory_file(path)
        if content:
            sections.append(f"# {name}\n{content}")

    return {
        "memory_context": "\n\n".join(sections) or "No memory context found."
    }

def route_after_interpretation(state: AgentV2State) -> str:
    """
    Decide whether context retrieval is required.

    Retrieval is skipped for requests that can be handled directly:
    - out of scope
    - product advice
    - unclear requests
    - no required external sources
    """

    interpreted = state.get("interpreted_request")

    if interpreted is None:
        return "generate_answer_draft"

    if interpreted.intent in {
        Intent.OUT_OF_SCOPE,
        Intent.PRODUCT_ADVICE,
        Intent.UNCLEAR,
    }:
        return "generate_answer_draft"

    if not interpreted.required_sources:
        return "generate_answer_draft"

    return "retrieve_context"


def make_generate_answer_node(model_name: str):

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    structured_llm = llm.with_structured_output(AgentAnswer)

    def generate_answer_draft(state: AgentV2State):

        interpreted = state["interpreted_request"]
        
        if interpreted.intent == Intent.PRODUCT_ADVICE:
            return {
                "answer_draft": AgentAnswer(
                    mode=ResponseMode.OUT_OF_SCOPE,
                    answer=(
                        "Je ne peux pas recommander un produit structuré à vendre ou à pousser commercialement. "
                        "En revanche, je peux aider à cadrer la faisabilité de référencement si tu me donnes "
                        "un assureur cible, un wrapper, un émetteur pressenti, une maturité, un payoff et les sous-jacents. "
                        "Je peux aussi résumer les contraintes d’un assureur pour identifier les structures généralement "
                        "plus simples ou plus difficiles à faire référencer, sans faire de recommandation commerciale."
                    ),
                    sources_used=[],
                    missing_information=[
                        "assureur ou plateforme cible",
                        "wrapper",
                        "émetteur pressenti",
                        "maturité",
                        "payoff",
                        "sous-jacents",
                    ],
                    confidence="high",
        )
    }

        try:

            result = structured_llm.invoke(
                [
                    {
                        "role": "user",
                        "content": (
                            "INTERPRETED REQUEST:\n"
                            f"{interpreted.model_dump_json(indent=2)}\n\n"
                            "MEMORY CONTEXT:\n"
                            f"{state.get('memory_context', '')}\n\n"
                            "CONTEXT SUMMARY:\n"
                            f"{state.get('context_summary', '')}\n\n"
                            "RETRIEVED CONTEXT:\n"
                            f"{state.get('retrieved_context', [])}"
                        ),
                    },
                ]
            )

            return {
                "answer_draft": result
            }

        except Exception as e:

            return {
                "error": str(e)
            }

    return generate_answer_draft



def retrieve_context(state: AgentV2State) -> AgentV2State:
    """
    Retrieve context from the local multi-source vector index.

    The runtime agent does not read raw files directly.
    It only consumes the offline-built vector index from data_agent_vect.
    """

    interpreted = state.get("interpreted_request")

    if interpreted is None:
        return {
            "retrieved_context": [],
            "error": "No interpreted_request available for context retrieval.",
        }

    source_types = [source.value for source in interpreted.required_sources]

    if not source_types:
        return {
            "retrieved_context": [],
        }

    try:
        retrieved_context = retrieve_context_from_index(
            query=state["user_query"],
            insurers=interpreted.insurers,
            source_types=source_types,
            k_per_source=3,
        )

        return {
            "retrieved_context": retrieved_context,
        }

    except Exception as exc:
        return {
            "retrieved_context": [],
            "error": str(exc),
        }


def summarize_retrieved_context(state: AgentV2State) -> AgentV2State:
    """
    Build a compact summary of retrieved context before answer generation.

    This is not a rule engine.
    It helps the final LLM distinguish:
    - issuer rules
    - underlying rules
    - wrapper rules
    - ESG constraints
    - validation requirements
    - missing or weak sources

    It uses normalized retrieval output when available:
    - valid_retrieved_context
    - missing_context_sources
    - weak_context_sources
    """

    retrieved_context = state.get(
        "valid_retrieved_context",
        state.get("retrieved_context", []),
    )

    missing_sources = state.get("missing_context_sources", [])
    weak_sources = state.get("weak_context_sources", [])

    lines = []

    if retrieved_context:
        lines.append("Retrieved usable context:")

    for item in retrieved_context:
        entity = item.get("entity") or "UNKNOWN_ENTITY"
        source_type = item.get("source_type") or "UNKNOWN_SOURCE_TYPE"
        source_name = item.get("source_name") or "UNKNOWN_SOURCE"
        page = item.get("page")
        content = (item.get("content") or "").strip()

        if not content:
            continue

        header = f"{entity} | {source_type} | {source_name}"
        if page is not None:
            header += f" | page {page}"

        content_lower = content.lower()
        detected_topics = []

        topic_rules = {
            "issuer rules": [
                "émetteur",
                "emetteur",
                "issuer",
                "contrepartie",
                "counterparty",
                "rating",
                "notation",
            ],
            "underlying rules": [
                "sous-jacent",
                "sous-jacents",
                "underlying",
                "underlyings",
                "indice",
                "indices",
                "panier",
                "basket",
                "titre",
                "actions",
            ],
            "wrapper rules": [
                "uc",
                "unit-linked",
                "unité de compte",
                "unites de compte",
                "fonds euro",
                "assurance vie",
                "capitalisation",
            ],
            "ESG constraints": [
                "esg",
                "exclusion",
                "exclusions",
                "controverse",
                "controverses",
            ],
            "validation requirement": [
                "validation",
                "pré-validation",
                "pre-validation",
                "agrément",
                "agrement",
                "accord préalable",
                "cas par cas",
                "soumis à validation",
            ],
            "maturity constraints": [
                "maturité",
                "maturite",
                "tenor",
                "durée",
                "duree",
                "ans",
                "années",
                "annees",
            ],
            "refusal / exclusion rules": [
                "refusé",
                "refuse",
                "refus",
                "non-référençable",
                "non referencable",
                "exclu",
                "exclus",
                "interdit",
            ],
        }

        for topic, keywords in topic_rules.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_topics.append(topic)

        if not detected_topics:
            detected_topics.append("general context")

        excerpt = " ".join(content.split())
        if len(excerpt) > 700:
            excerpt = excerpt[:700] + "..."

        lines.append(
            f"- {header}: topics={detected_topics}. Excerpt: {excerpt}"
        )

    if missing_sources:
        lines.append("")
        lines.append("Missing context sources:")
        for source in missing_sources:
            entity = source.get("entity") or "UNKNOWN_ENTITY"
            source_type = source.get("source_type") or "UNKNOWN_SOURCE_TYPE"
            message = source.get("message") or "No relevant context retrieved."
            lines.append(
                f"- {entity} | {source_type}: {message}"
            )

    if weak_sources:
        lines.append("")
        lines.append("Weak context sources:")
        for source in weak_sources:
            entity = source.get("entity") or "UNKNOWN_ENTITY"
            source_type = source.get("source_type") or "UNKNOWN_SOURCE_TYPE"
            source_name = source.get("source_name") or "UNKNOWN_SOURCE"
            page = source.get("page")
            reason = source.get("reason") or "Weak or administrative context."

            header = f"{entity} | {source_type} | {source_name}"
            if page is not None:
                header += f" | page {page}"

            lines.append(
                f"- {header}: {reason}"
            )

    if not lines:
        return {
            "context_summary": "No relevant context was retrieved."
        }

    return {
        "context_summary": "\n".join(lines)
    }


def make_judge_answer_node(model_name: str = "gpt-4.1"):
    """
    Create a final answer judge node.
    The judge reviews the draft answer and either keeps it or corrects it.
    """

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    structured_llm = llm.with_structured_output(AgentAnswer)

    def judge_answer(state: AgentV2State) -> AgentV2State:
        answer_draft = state.get("answer_draft")
        interpreted = state.get("interpreted_request")

        if answer_draft is None:
            return {
                "error": "No answer_draft available for judge.",
            }

        try:
            result = structured_llm.invoke(
                [
                    {
                        "role": "system",
                        "content": JUDGE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": (
                            "INTERPRETED REQUEST:\n"
                            f"{interpreted.model_dump_json(indent=2) if interpreted else None}\n\n"
                            "MEMORY CONTEXT:\n"
                            f"{state.get('memory_context', '')}\n\n"
                            "CONTEXT SUMMARY:\n"
                            f"{state.get('context_summary', '')}\n\n"
                            "RETRIEVED CONTEXT:\n"
                            f"{state.get('retrieved_context', [])}\n\n"
                            "DRAFT ANSWER:\n"
                            f"{answer_draft.model_dump_json(indent=2)}"
                        ),
                    },
                ]
            )

            return {
                "final_answer": result
            }

        except Exception as exc:
            return {
                "final_answer": answer_draft,
                "error": f"Judge failed, using draft answer: {exc}",
            }

    return judge_answer


def analyze_evidence_node(state: AgentV2State) -> AgentV2State:
    interpreted = state["interpreted_request"]
    contexts = state.get("retrieved_context", [])

    # règle simple V1 : classifier les chunks récupérés
    evidence_by_insurer = {}

    for insurer in interpreted.insurers:
        insurer_contexts = [
            c for c in contexts
            if c.get("entity", "").lower() == insurer.lower()
        ]

        rules = []

        if any("émetteur" in c.get("content", "").lower() or "issuer" in c.get("content", "").lower()
               for c in insurer_contexts):
            rules.append(RuleEvidence(
                entity=insurer,
                rule_type="issuer",
                status=EvidenceStatus.SUPPORTED,
                evidence=[],
                finding="Issuer rules found in retrieved context.",
            ))
        else:
            rules.append(RuleEvidence(
                entity=insurer,
                rule_type="issuer",
                status=EvidenceStatus.MISSING,
                finding="No explicit issuer rule found.",
                limitation="Issuer eligibility cannot be assessed from retrieved context.",
            ))

        if any("sous-jacent" in c.get("content", "").lower() or "underlying" in c.get("content", "").lower()
               for c in insurer_contexts):
            rules.append(RuleEvidence(
                entity=insurer,
                rule_type="underlying",
                status=EvidenceStatus.SUPPORTED,
                evidence=[],
                finding="Underlying rules found in retrieved context.",
            ))
        else:
            rules.append(RuleEvidence(
                entity=insurer,
                rule_type="underlying",
                status=EvidenceStatus.MISSING,
                finding="No explicit underlying rule found.",
                limitation="Underlying eligibility cannot be assessed from retrieved context.",
            ))

        evidence_by_insurer[insurer] = rules

    state["evidence_analysis"] = EvidenceAnalysis(
        by_insurer=evidence_by_insurer,
        global_limitations=[],
        confidence="medium",
    )
    return state


def normalize_retrieved_context(state: AgentV2State) -> AgentV2State:
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