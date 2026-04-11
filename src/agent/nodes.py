# src/agent/nodes.py

from __future__ import annotations

import re
from typing import List

from src.domain.models import (
    ProductRequest,
    FinalAnswer,
    ResponseMode,
    UserIntent,
)

import unicodedata

# =========================
# KNOWN INSURERS DICTIONARY
# =========================

KNOWN_INSURERS = [
    "swisslife",
    "abeille",
    "cnp",
    "axa",
    "generali",
    "générali",
]

"""
List of insurers currently recognized by the heuristic parser.

This list supports early-stage parsing before introducing
LLM-based structured extraction.

Future versions should replace this with:
- taxonomy-driven lookup
- metadata-based recognition
- or vector-based entity detection
"""

# =========================
# UTILITY FUNCTIONS
# =========================

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    # apostrophes typographiques → apostrophe simple
    text = text.replace("’", "'").replace("‘", "'")
    # tirets spéciaux → tiret simple
    text = text.replace("–", "-").replace("—", "-")
    # espaces multiples
    text = " ".join(text.split())
    return text

# =========================
# HEURISTIC ENTITY DETECTION FUNCTIONS
# =========================

def _detect_insurers(text: str) -> List[str]:
    """
    Detect insurers mentioned in the user query.

    Returns a deduplicated list while preserving original order.
    """

    lower = normalize_text(text)
    found = []

    if "swisslife" in lower:
        found.append("SwissLife")

    if "abeille" in lower:
        found.append("Abeille")

    if "cnp" in lower:
        found.append("CNP")

    if "axa" in lower:
        found.append("Axa")

    if "generali" in lower or "générali" in lower:
        found.append("Generali")

    # Remove duplicates while preserving insertion order
    seen = set()
    ordered = []

    for insurer in found:
        if insurer not in seen:
            ordered.append(insurer)
            seen.add(insurer)

    return ordered


def _detect_product_type(text: str) -> str | None:
    """
    Detect structured product family.

    Examples:
    - autocall
    - phoenix

    Should later be replaced by taxonomy-driven classification.
    """

    lower = normalize_text(text)    

    if "autocall" in lower:
        return "autocall"

    if "callable note" in lower:
        return "callable_note"

    return None


def _detect_payoff_type(text: str) -> str | None:
    """
    Detect payoff mechanics.

    Examples:
    - worst-of
    - best-of
    """

    lower = normalize_text(text)

    if "worst-of" in lower or "worst of" in lower:
        return "worst-of"

    if "decrement" in lower or "best of" in lower:
        return "decrement"

    return None


def _detect_underlying_type(text: str) -> str | None:
    """
    Detect underlying asset category.

    Examples:
    - single_stock
    - index
    - basket
    """

    lower = normalize_text(text)

    if "single stock" in lower or "single stocks" in lower:
        return "single_stock"

    if "indice" in lower or "index" in lower:
        return "index"

    if "worst-of" in lower or "basket" in lower:
        return "worst-of"

    return None


def _detect_features(text: str) -> List[str]:
    """
    Detect structural product features.

    Examples:
    - decrement
    - memory coupon
    - worst-of mechanism
    """

    lower = normalize_text(text)
    features = []

    if "décrément" in lower or "decrement" in lower:
        features.append("decrement")

    if "memory" in lower:
        features.append("memory_coupon")

    if "worst-of" in lower or "worst of" in lower:
        features.append("worst_of_mechanism")

    return features


# =========================
# NODE: PARSE USER QUERY
# =========================

def parse_user_query(state):
    """
    Extract structured product information from the raw user query.

    This node converts natural-language input into a ProductRequest object.

    Current implementation:
    heuristic keyword-based extraction

    Future implementation:
    LLM structured extraction with JSON schema validation
    """

    user_query = state["user_query"]

    parsed = ProductRequest(
        insurers=_detect_insurers(user_query),
        product_type=_detect_product_type(user_query),
        payoff_type=_detect_payoff_type(user_query),
        underlying_type=_detect_underlying_type(user_query),
        underlyings=[],
        features=_detect_features(user_query),
        # issuer="Goldman Sachs"
        # if "goldman sachs" in user_query.lower()
        # else None,
        # capital_protection=None,
        # maturity_years=None,
        # raw_user_query=user_query,
    )

    return {
        "parsed_request": parsed,
    }


# =========================
# NODE: CLASSIFY USER INTENT
# =========================

def classify_user_intent(state):
    """
    Classify the user's intent.

    Determines which workflow branch should be executed:

    - decision workflow
    - constraint summary workflow
    - multi-insurer comparison workflow
    - clarification workflow
    """

    req = state["parsed_request"]
    text = normalize_text(state["user_query"])

    # Multi-insurer detection
    if len(req.insurers) >= 2:
        return {"intent": UserIntent.COMPARISON_REQUEST}

    summary_markers = [
        "quelles sont les contraintes",
        "quels sont les points de vigilance",
        "qu'est-ce qu'il faut savoir",
        "que faut-il savoir",
        "synthèse",
        "résumé",
        "resume",
    ]

    if any(marker in text for marker in summary_markers):
        return {"intent": UserIntent.CONSTRAINT_SUMMARY}

    decision_markers = [
        "est-ce que c'est possible",
        "est-ce possible",
        "est-ce que je peux",
        "puis-je",
        "référençable",
        "referencable",
    ]

    if any(marker in text for marker in decision_markers):
        return {"intent": UserIntent.DECISION_REQUEST}

    # Default fallback logic:
    # If insurer exists but product description incomplete → summary mode
    if req.insurers and not req.product_type:
        return {"intent": UserIntent.CONSTRAINT_SUMMARY}

    return {"intent": UserIntent.CLARIFICATION_NEEDED}


# =========================
# NODE: DETECT MISSING INFORMATION
# =========================

def detect_missing_information(state):
    """
    Identify missing fields required to perform feasibility analysis.

    Only applies to decision-oriented workflows.
    """

    req = state["parsed_request"]
    intent = state["intent"]

    missing = []

    if not req.insurers:
        missing.append("assureur")

    if intent in {
        UserIntent.DECISION_REQUEST,
        UserIntent.COMPARISON_REQUEST,
    }:
        if not req.product_type:
            missing.append("type de produit")

        if not req.underlying_type:
            missing.append("type de sous-jacent")

    return {"missing_fields": missing}


# =========================
# NODE: ROUTING AFTER ANALYSIS
# =========================

def route_after_analysis(state):
    """
    Routing node used by LangGraph conditional edges.

    Selects which response generator node should be executed.
    """

    intent = state["intent"]
    missing = state.get("missing_fields", [])

    if intent == UserIntent.CONSTRAINT_SUMMARY:
        return "generate_summary_answer"

    if intent in {
        UserIntent.DECISION_REQUEST,
        UserIntent.COMPARISON_REQUEST,
    }:
        if missing:
            return "generate_partial_answer"

        return "generate_decision_stub"

    return "generate_partial_answer"


# =========================
# NODE: SUMMARY RESPONSE GENERATION
# =========================

def generate_summary_answer(state):
    """
    Generate a high-level insurer constraint summary.

    Used when the request is exploratory or incomplete.
    """

    req = state["parsed_request"]

    insurers_txt = (
        ", ".join(req.insurers)
        if req.insurers
        else "the requested insurer"
    )

    answer = FinalAnswer(
        mode=ResponseMode.SUMMARY,
        summary=(
            f"Summary of key referencing constraints to review for "
            f"{insurers_txt}: verify eligible underlyings, exclusions on "
            "single stocks, restrictions on proprietary indices, presence "
            "of decrement mechanisms, and structural characteristics such "
            "as autocall or phoenix payoff types."
        ),
        key_constraints=[
            "Underlying eligibility rules",
            "Presence of decrement mechanisms",
            "Proprietary index restrictions",
            "Structured payoff type constraints",
        ],
        missing_information=[],
        sources=[],
        confidence="low",
    )

    return {"final_answer": answer}


# =========================
# NODE: PARTIAL RESPONSE GENERATION
# =========================

def generate_partial_answer(state):
    """
    Generate a preliminary response when information is insufficient
    for a full feasibility decision.
    """

    req = state["parsed_request"]
    missing = state.get("missing_fields", [])

    insurers_txt = (
        ", ".join(req.insurers)
        if req.insurers
        else "the requested insurer"
    )

    answer = FinalAnswer(
        mode=ResponseMode.PARTIAL,
        summary=(
            f"A preliminary referencing analysis can be performed for "
            f"{insurers_txt}, but additional information is required "
            "to reach a definitive conclusion."
        ),
        key_constraints=[
            "Underlying eligibility constraints",
            "Structured payoff mechanics",
            "Decrement / proprietary index restrictions",
        ],
        missing_information=missing,
        sources=[],
        confidence="medium",
    )

    return {"final_answer": answer}


# =========================
# NODE: DECISION STUB GENERATION
# =========================

def generate_decision_stub(state):
    """
    Placeholder feasibility response.

    This node confirms that the decision workflow executes correctly
    before connecting the rule-extraction engine and RAG pipeline.
    """

    req = state["parsed_request"]

    insurers_txt = (
        ", ".join(req.insurers)
        if req.insurers
        else "the requested insurer"
    )

    product_bits = [
        req.product_type,
        req.payoff_type,
        req.underlying_type,
    ]

    product_desc = " / ".join(
        [x for x in product_bits if x]
    )

    answer = FinalAnswer(
        mode=ResponseMode.DECISION,
        summary=(
            f"Preliminary feasibility workflow executed successfully for "
            f"{insurers_txt} on product configuration "
            f"{product_desc or 'provided by the user'}. "
            "Document-based rule evaluation is not yet connected."
        ),
        key_constraints=[
            "Document-level rule engine not yet connected",
        ],
        missing_information=[],
        sources=[],
        confidence="low",
    )

    return {"final_answer": answer}