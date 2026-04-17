from langgraph.graph import StateGraph, START, END

from src.agent.state_chatbot import ChatbotState
from src.agent.nodes_chatbot import make_generate_chatbot_answer


def build_chatbot_graph(model_name: str):
    """
    Build and compile a minimal LangGraph workflow for the chatbot prototype.

    The graph is intentionally minimal:
        user question
        -> LLM answer

    The model name is injected at graph construction time so that
    different models can be tested without changing the workflow logic.
    """

    builder = StateGraph(ChatbotState)

    generate_chatbot_answer = make_generate_chatbot_answer(model_name=model_name)

    builder.add_node("generate_chatbot_answer", generate_chatbot_answer)

    builder.add_edge(START, "generate_chatbot_answer")
    builder.add_edge("generate_chatbot_answer", END)

    return builder.compile()