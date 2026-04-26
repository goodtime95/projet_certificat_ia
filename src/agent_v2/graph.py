from langgraph.graph import StateGraph, END

from src.agent_v2.state import AgentV2State
from src.agent_v2.nodes import (
    make_interpret_user_request_node,
    make_generate_answer_node,
    retrieve_context,
)


def build_agent_v2_graph(model_name: str):

    graph = StateGraph(AgentV2State)

    graph.add_node(
        "interpret_user_request",
        make_interpret_user_request_node(model_name),
    )

    graph.add_node(
        "retrieve_context",
        retrieve_context,
    )

    graph.add_node(
        "generate_answer",
        make_generate_answer_node(model_name),
    )

    graph.set_entry_point("interpret_user_request")

    graph.add_edge(
        "interpret_user_request",
        "retrieve_context",
    )

    graph.add_edge(
        "retrieve_context",
        "generate_answer",
    )

    graph.add_edge(
        "generate_answer",
        END,
    )

    return graph.compile()