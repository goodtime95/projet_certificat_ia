from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class Intent(str, Enum):
    REFERENCING_FEASIBILITY = "referencing_feasibility"
    CONSTRAINT_SUMMARY = "constraint_summary"
    POLICY_CONFIRMATION = "policy_confirmation"
    MEMORY_OR_HISTORY = "memory_or_history"
    PRODUCT_ADVICE = "product_advice"
    OUT_OF_SCOPE = "out_of_scope"
    UNCLEAR = "unclear"

class ScopeStatus(str, Enum):
    IN_SCOPE = "in_scope"
    NEEDS_CLARIFICATION = "needs_clarification"
    OUT_OF_SCOPE = "out_of_scope"

class ResponseMode(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    REJECT_INCORRECT_PREMISE = "reject_incorrect_premise"
    OUT_OF_SCOPE = "out_of_scope"

class SourceNeed(str, Enum):
    REFERENCING_CHARTER = "referencing_charter"
    EMAIL_HISTORY = "email_history"
    PRODUCT_DOCUMENTATION = "product_documentation"
    INTERNAL_NOTE = "internal_note"
    USER_MEMORY = "user_memory"
    NONE = "none"

class ProductCandidate(BaseModel):
    label: Optional[str] = None
    product_type: Optional[str] = None
    payoff_type: Optional[str] = None
    underlying_type: Optional[str] = None
    underlyings: List[str] = Field(default_factory=list)
    maturity: Optional[str] = None
    issuer: Optional[str] = None
    wrapper: Optional[str] = None
    features: List[str] = Field(default_factory=list)

class DetectedInconsistency(BaseModel):
    code: str
    message: str

class InterpretedRequest(BaseModel):
    intent: Intent
    insurers: List[str] = Field(default_factory=list)
    products: List[ProductCandidate] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    detected_inconsistencies: List[DetectedInconsistency] = Field(default_factory=list)
    scope_status: ScopeStatus
    user_needs_documents: bool = True
    required_sources: List[SourceNeed] = Field(default_factory=list)

class SourceReference(BaseModel):
    source_type: str
    entity: Optional[str]
    source_name: Optional[str]
    page: Optional[int]


class AgentAnswer(BaseModel):
    mode: ResponseMode
    answer: str
    missing_information: list[str] = []
    sources_used: list[SourceReference] = []
    confidence: Literal[
        "low",
        "medium",
        "high"
    ] = "medium"

class ContextRoute(str, Enum):
    USE_CONTEXT = "use_context"
    SKIP_CONTEXT = "skip_context"


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    WEAK = "weak"
    MISSING = "missing"
    CONTRADICTED = "contradicted"


class RuleEvidence(BaseModel):
    entity: str
    rule_type: str  # issuer | underlying | wrapper | validation | esg | currency
    status: EvidenceStatus
    evidence: list[SourceReference] = []
    finding: str
    limitation: str | None = None


class EvidenceAnalysis(BaseModel):
    by_insurer: dict[str, list[RuleEvidence]]
    global_limitations: list[str] = []
    confidence: str  # high | medium | low


