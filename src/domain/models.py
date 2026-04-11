from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field



# =========================
# USER INTENT ENUMERATION
# =========================

class UserIntent(str, Enum):
    """
    Detected intent from the user's request.

    This drives routing inside the LangGraph workflow:
    - feasibility analysis
    - insurer constraint summary
    - multi-insurer comparison
    - clarification request if input is incomplete
    """

    DECISION_REQUEST = "decision_request"
    # User explicitly asks whether a product can be listed / referenced

    CONSTRAINT_SUMMARY = "constraint_summary"
    # User requests a high-level overview of insurer constraints

    COMPARISON_REQUEST = "comparison_request"
    # User requests analysis across multiple insurers

    CLARIFICATION_NEEDED = "clarification_needed"
    # Request too vague or incomplete for a decision


# =========================
# DECISION STATUS ENUMERATION
# =========================

class DecisionStatus(str, Enum):
    """
    Possible outcomes of the referencing feasibility analysis
    for a given insurer.
    """

    ALLOWED = "allowed"
    # Product is compatible with known insurer constraints

    FORBIDDEN = "forbidden"
    # Product is explicitly prohibited

    CONDITIONAL = "conditional"
    # Product is allowed only under specific conditions

    INSUFFICIENT_INFO = "insufficient_info"
    # Not enough information to conclude


# =========================
# RESPONSE MODE ENUMERATION
# =========================

class ResponseMode(str, Enum):
    """
    Global response type returned by the agent.

    Determines how the response should be interpreted
    by downstream systems (UI, API consumers, etc.).
    """

    DECISION = "decision"
    # Structured feasibility decision per insurer

    SUMMARY = "summary"
    # High-level summary of insurer constraints

    PARTIAL = "partial"
    # Preliminary analysis with missing information identified


# =========================
# STRUCTURED PRODUCT REQUEST
# =========================

class ProductRequest(BaseModel):
    """
    Structured representation of the user's request.

    Produced by the parsing layer (heuristics or LLM extraction).
    Used as the main input for rule retrieval and feasibility analysis.
    """

    insurers: List[str] = Field(default_factory=list)
    # Target insurers (e.g. CNP, AXA, Generali)

    product_type: Optional[str] = None
    # Structured product family (autocall, phoenix, reverse convertible, etc.)

    payoff_type: Optional[str] = None
    # Payoff mechanics (worst-of, best-of, vanilla, etc.)

    underlying_type: Optional[str] = None
    # Underlying category (index, single_stock, basket, proprietary_index, etc.)

    underlyings: List[str] = Field(default_factory=list)
    # Explicit underlying names if available (e.g. Eurostoxx 50, GS Electrification Index)

    features: List[str] = Field(default_factory=list)
    # Structural features (decrement, memory coupon, snowball, callable barrier, etc.)

    # issuer: Optional[str] = None
    # # Issuing bank (Goldman Sachs, BNP Paribas, Citi, etc.)

    # capital_protection: Optional[str] = None
    # # Capital protection level (100%, 90%, none, conditional)

    # maturity_years: Optional[float] = None
    # # Product maturity in years (relevant for certain insurer policies)

    # raw_user_query: str
    # # Original user input preserved for traceability


# =========================
# DOCUMENT-LEVEL CONSTRAINT RULE
# =========================

class ConstraintRule(BaseModel):
    """
    Rule extracted from insurer documentation
    (referencing charters, compliance notes, internal policies).

    These rules feed the decision engine.
    """

    insurer: str
    # Insurer to which the rule applies

    rule_type: str
    # allowed / forbidden / conditional

    scope: str
    # Rule scope (e.g. single_stock_decrement, proprietary_index, autocall_structure)

    condition: Optional[str] = None
    # Optional condition attached to the rule
    # Example: allowed only if UCITS-compliant index

    source_doc: str
    # Source document identifier (PDF name or internal reference)

    source_page: Optional[int] = None
    # Page number in the source document

    source_excerpt: str
    # Exact supporting text extracted from the document

    priority: int = 0
    # Rule priority level (used when resolving conflicts between sources)


# =========================
# DECISION RESULT PER INSURER
# =========================

class DecisionResult(BaseModel):
    """
    Feasibility result for a single insurer.

    Enables structured multi-insurer comparison output.
    """

    insurer: str
    # Insurer analyzed

    decision: DecisionStatus
    # Final feasibility decision

    rationale: List[str] = Field(default_factory=list)
    # Human-readable reasoning steps supporting the decision

    supporting_rules: List[ConstraintRule] = Field(default_factory=list)
    # Rules used to justify the decision

    missing_fields: List[str] = Field(default_factory=list)
    # Missing information preventing a definitive decision

    confidence: str = "medium"
    # Confidence level (low / medium / high)


# =========================
# FINAL AGENT RESPONSE STRUCTURE
# =========================

class FinalAnswer(BaseModel):
    """
    Standard output structure returned by the agent.

    Acts as a contract between:
    - LangGraph workflow
    - UI layer
    - future API endpoints
    """

    mode: ResponseMode
    # Response type (decision / summary / partial)

    summary: str
    # Human-readable explanation for sales users

    decisions: List[DecisionResult] = Field(default_factory=list)
    # Structured decisions per insurer

    key_constraints: List[str] = Field(default_factory=list)
    # Main constraints identified during analysis

    missing_information: List[str] = Field(default_factory=list)
    # Additional information required for completion

    sources: List[str] = Field(default_factory=list)
    # Referenced documentation sources (charters, internal notes, etc.)

    confidence: str = "medium"
    # Overall response confidence level