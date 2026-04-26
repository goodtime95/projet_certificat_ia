from typing import TypedDict, Any, Dict, List
from src.agent_v2.schemas import InterpretedRequest, AgentAnswer



class AgentV2State(TypedDict, total=False):

    user_query: str

    interpreted_request: InterpretedRequest

    retrieved_context: List[Dict[str, Any]]
    # Retrieved context from one or more available knowledge sources.
    # Current implementation supports local referencing charter files.
    # Future sources may include emails, product documentation, internal notes,
    # and user memory.

    answer: AgentAnswer

    model_used: str

    error: str