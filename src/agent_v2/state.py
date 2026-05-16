from typing import TypedDict, Any, Dict, List
from src.agent_v2.schemas import InterpretedRequest, AgentAnswer


class AgentV2State(TypedDict, total=False):

    user_query: str
    conversation_history: list[dict]

    interpreted_request: InterpretedRequest

    memory_context: str

    retrieved_context: List[Dict[str, Any]]
    missing_context_sources: List[Dict[str, Any]]

    context_route: str

    answer: AgentAnswer

    model_used: str

    error: str