from typing import Any, Dict, List, Optional, TypedDict
from src.domain.models import (
    ProductRequest,
    ConstraintRule,
    DecisionResult,
    FinalAnswer,
    UserIntent,
)
from src.domain.features import ProductFeatures


class AgentState(TypedDict, total=False):
    """
    Shared state object used across all LangGraph workflow nodes.

    Each node in the graph reads from and writes to this structure.
    It represents the evolving context of the agent during execution.

    The state progressively moves through these phases:

        user_query
            ↓
        parsed_request
            ↓
        intent classification
            ↓
        missing information detection
            ↓
        product_features
            ↓
        document retrieval (RAG)
            ↓
        constraint extraction
            ↓
        decision computation
            ↓
        final structured response

    Fields are optional (total=False) because they are populated
    incrementally by different nodes in the pipeline.
    """

    user_query: str
    # Raw user input string.
    # This is the original natural-language request submitted by the sales user.

    parsed_request: ProductRequest
    # Structured representation of the user request produced by the parsing node.
    # Includes insurers, product type, underlying type, structural features, etc.

    intent: UserIntent
    # Classified intent of the request.
    # Determines which workflow branch the graph should execute:
    # decision analysis, constraint summary, comparison, or clarification.

    missing_fields: List[str]
    # List of missing information required to complete feasibility analysis.
    # Example:
    # ["underlying_type", "capital_protection"]

    product_features: ProductFeatures
    # Normalized business representation derived from parsed_request.
    # This object isolates rule-relevant product characteristics in a
    # deterministic format that can later be consumed by the rule engine.
    # Example:
    # - has_autocall
    # - has_worst_of
    # - underlying_type
    # - basket_size

    retrieved_chunks: List[Dict[str, Any]]
    # Document fragments retrieved from the RAG layer.
    # Typically includes:
    # - text content
    # - metadata (insurer, page number, section)
    # - source document identifiers

    extracted_rules: List[ConstraintRule]
    # Structured rules extracted from retrieved documentation.
    # Example:
    # "single-stock decrement structures are not allowed"

    decisions: List[DecisionResult]
    # Per-insurer feasibility decisions produced by the rule engine.
    # Supports multi-insurer comparison workflows.

    final_answer: FinalAnswer
    # Final structured response returned by the agent.
    # Contains:
    # - response mode (decision / summary / partial)
    # - explanation
    # - supporting rules
    # - missing information
    # - confidence score

    error: str
    # Optional error message captured during execution.
    # Useful for:
    # - debugging
    # - logging
    # - safe failure handling in production pipelines