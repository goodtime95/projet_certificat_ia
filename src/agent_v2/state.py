from typing import TypedDict, Any, Dict, List
from src.agent_v2.schemas import InterpretedRequest, AgentAnswer



class AgentV2State(TypedDict, total=False):

    user_query: str

    interpreted_request: InterpretedRequest

    retrieved_chunks: List[Dict[str, Any]]

    answer: AgentAnswer

    model_used: str

    error: str