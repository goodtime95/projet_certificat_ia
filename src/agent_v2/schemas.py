from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class Intent(str, Enum):
    REFERENCING_FEASIBILITY = "referencing_feasibility"
    CONSTRAINT_SUMMARY = "constraint_summary"
    POLICY_CONFIRMATION = "policy_confirmation"
    MEMORY_OR_HISTORY = "memory_or_history"
    PRODUCT_ADVICE = "product_advice"
    END_CONVERSATION = "end_conversation"
    RESET_CONVERSATION = "reset_conversation"
    OUT_OF_SCOPE = "out_of_scope"
    UNCLEAR = "unclear"

class SourceNeed(str, Enum):
    REFERENCING_CHARTER = "referencing_charter"
    EMAIL_HISTORY = "email_history"
    PRODUCT_DOCUMENTATION = "product_documentation"
    INTERNAL_NOTE = "internal_note"
    USER_MEMORY = "user_memory"

class ProductCandidate(BaseModel):
    label: Optional[str] = None
    product_type: Optional[str] = None
    payoff_type: Optional[str] = None
    underlying_type: Optional[str] = None
    underlyings: List[str] = Field(default_factory=list)
    maturity: Optional[str] = None
    issuer: Optional[str] = None
    features: List[str] = Field(default_factory=list)

class DetectedInconsistency(BaseModel):
    code: str
    message: str

class InterpretedRequest(BaseModel):
    standalone_query: Optional[str] = None
    intent: Intent
    insurers: List[str] = Field(default_factory=list)
    products: List[ProductCandidate] = Field(default_factory=list)
    detected_inconsistencies: List[DetectedInconsistency] = Field(default_factory=list)
    required_sources: List[SourceNeed] = Field(default_factory=list)
    wants_raw_sources: bool = False
    source_output_level: Literal["names_only", "excerpts", "full_raw"] = "excerpts"

class RawSourceOutput(BaseModel):
    source_id: str
    entity: str | None = None
    source_type: str
    source_name: str
    page: int | None = None
    raw_text: str | None = None
    excerpt: str | None = None

class ContextRoute(str, Enum):
    USE_CONTEXT = "use_context"
    SKIP_CONTEXT = "skip_context"

class ResponseMode(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    REJECT_INCORRECT_PREMISE = "reject_incorrect_premise"
    OUT_OF_SCOPE = "out_of_scope"
    RAW_SOURCES = "raw_sources"

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
    source_ids: list[str] = []
    raw_sources: list[RawSourceOutput] = []
    confidence: Literal[
        "low",
        "medium",
        "high"
    ] = "medium"

class ConversationContext(BaseModel):
    active_insurers: List[str] = Field(default_factory=list)
    active_products: List[ProductCandidate] = Field(default_factory=list)
    active_topic: Optional[str] = None
    last_intent: Optional[Intent] = None






