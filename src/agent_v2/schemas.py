from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Intent(str, Enum):
    REFERENCING_FEASIBILITY = "referencing_feasibility"
    REFERENCING_COMPARISON = "referencing_comparison"
    CONSTRAINT_SUMMARY = "constraint_summary"
    POLICY_CONFIRMATION = "policy_confirmation"
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


class AgentAnswer(BaseModel):

    mode: ResponseMode

    summary: str

    next_steps: List[str] = Field(default_factory=list)

    warnings: List[str] = Field(default_factory=list)