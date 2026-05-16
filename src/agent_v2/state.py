from typing import TypedDict, Any, Dict, List
from src.agent_v2.schemas import InterpretedRequest, AgentAnswer, ConversationContext


class AgentV2State(TypedDict, total=False):

    user_query: str

    conversation_context: ConversationContext

    interpreted_request: InterpretedRequest

    memory_context: str

    retrieved_context: List[Dict[str, Any]]
    missing_context_sources: List[Dict[str, Any]]

    retrieval_status: Dict[str, Any]

    context_route: str

    answer: AgentAnswer
    last_answer: AgentAnswer

    model_used: str

    error: str

    
