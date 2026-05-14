from typing import TypedDict, Any, Dict, List
from src.agent_v2.schemas import InterpretedRequest, AgentAnswer, EvidenceAnalysis



class AgentV2State(TypedDict, total=False):

    user_query: str

    interpreted_request: InterpretedRequest
    retrieved_context: List[Dict[str, Any]]
    # Retrieved context from one or more available knowledge sources.
    # Current implementation supports local referencing charter files.
    # Future sources may include emails, product documentation, internal notes,
    # and user memory.

    context_summary: str

    answer_draft: AgentAnswer
    final_answer: AgentAnswer
    context_route: str

    conversation_history: list[dict]
    memory_context: str

    evidence_analysis: EvidenceAnalysis | None = None

    valid_retrieved_context: List[Dict[str, Any]]
    missing_context_sources: List[Dict[str, Any]]
    weak_context_sources: List[Dict[str, Any]]

    model_used: str

    error: str