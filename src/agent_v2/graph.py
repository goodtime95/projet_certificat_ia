from langgraph.graph import StateGraph, START, END

from src.agent_v2.state import AgentV2State
from src.agent_v2.nodes import (
    make_interpret_user_request_node,
    route_after_interpretation,
    make_generate_fast_response_node,
    load_memory_context,
    route_after_memory,
    retrieve_context,
    make_generate_answer_node,
    route_after_retrieval,
    raw_sources_node,
  
)


def build_agent_v2_graph(model_name: str):

    graph = StateGraph(AgentV2State)

    graph.add_node(
        "interpret_user_request",
        make_interpret_user_request_node(model_name),
    )

    graph.add_node("generate_fast_response", 
                    make_generate_fast_response_node("gpt-4.1-mini")
    )

    graph.add_node(
        "load_memory_context",
        load_memory_context,
    )

    graph.add_node(
        "retrieve_context",
        retrieve_context,
    )

    graph.add_node("raw_sources_node", 
                   raw_sources_node
    )

    graph.add_node("generate_answer", 
                     make_generate_answer_node(model_name)
    )

    graph.add_edge(START, "interpret_user_request")

    graph.add_conditional_edges(
        "interpret_user_request",
        route_after_interpretation,
        {
            "generate_fast_response": "generate_fast_response",
            "load_memory_context": "load_memory_context",
            "raw_sources_node": "raw_sources_node",
        },
    )
    

    graph.add_conditional_edges(
        "load_memory_context",
        route_after_memory,
        {
            "retrieve_context": "retrieve_context",
            "generate_answer": "generate_answer",
        },
    )

    graph.add_conditional_edges(
        "retrieve_context",
        route_after_retrieval,
        {
            "raw_sources_node": "raw_sources_node",
            "generate_answer": "generate_answer",
        },
    )

    graph.add_edge("generate_fast_response", END)
    graph.add_edge("generate_answer", END)
    graph.add_edge("raw_sources_node", END)

    return graph.compile()