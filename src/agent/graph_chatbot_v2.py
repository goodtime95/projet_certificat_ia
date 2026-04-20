from langgraph.graph import END, START, StateGraph

from src.agent.nodes_chatbot import make_generate_structured_chatbot_answer
from src.agent.state_chatbot import ChatbotState


def build_chatbot_graph(model_name: str):
    """
    Build and compile a minimal LangGraph workflow for the structured chatbot prototype.
    """

    builder = StateGraph(ChatbotState)

    generate_structured_chatbot_answer = make_generate_structured_chatbot_answer(
        model_name=model_name
    )

    builder.add_node(
        "generate_structured_chatbot_answer",
        generate_structured_chatbot_answer,
    )

    builder.add_edge(START, "generate_structured_chatbot_answer")
    builder.add_edge("generate_structured_chatbot_answer", END)

    return builder.compile()