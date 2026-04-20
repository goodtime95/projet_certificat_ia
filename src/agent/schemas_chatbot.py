from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class UserIntent(str, Enum):
    REFERENCING_FEASIBILITY = "referencing_feasibility"
    REFERENCING_COMPARISON = "referencing_comparison"
    POLICY_CONFIRMATION = "policy_confirmation"
    OUT_OF_SCOPE = "out_of_scope"
    GENERAL_KNOWLEDGE = "general_knowledge"
    UNCLEAR = "unclear"


class ScopeStatus(str, Enum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    NEEDS_CLARIFICATION = "needs_clarification"


class ResponseMode(str, Enum):
    CLARIFY = "clarify"
    REJECT_INCORRECT_PREMISE = "reject_incorrect_premise"
    PRELIMINARY_DECISION = "preliminary_decision"
    OUT_OF_SCOPE = "out_of_scope"


class ProductRequest(BaseModel):
    insurers: List[str] = Field(default_factory=list)
    product_type: Optional[str] = None
    payoff_type: Optional[str] = None
    underlying_type: Optional[str] = None
    underlyings: List[str] = Field(default_factory=list)
    maturity: Optional[str] = None
    issuer: Optional[str] = None
    wrapper: Optional[str] = None
    features: List[str] = Field(default_factory=list)


class DetectedIssue(BaseModel):
    code: str
    message: str


class FinalAnswer(BaseModel):
    mode: ResponseMode
    summary: str


class StructuredChatbotOutput(BaseModel):
    intent: UserIntent
    parsed_request: ProductRequest
    missing_fields: List[str] = Field(default_factory=list)
    detected_inconsistencies: List[DetectedIssue] = Field(default_factory=list)
    scope_status: ScopeStatus
    final_answer: FinalAnswer