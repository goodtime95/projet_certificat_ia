from langgraph.graph import StateGraph, START, END

from src.agent_v2.state import AgentV2State
from src.agent_v2.nodes import (
    make_interpret_user_request_node,
    make_generate_answer_node,
    retrieve_context,
    summarize_retrieved_context,
    normalize_retrieved_context,
    route_after_interpretation,
    make_judge_answer_node,
    load_memory_context,
    analyze_evidence_node,
)


def build_agent_v2_graph(model_name: str):

    graph = StateGraph(AgentV2State)

    graph.add_node(
        "interpret_user_request",
        make_interpret_user_request_node(model_name),
    )

    graph.add_node(
        "load_memory_context",
        load_memory_context,
    )

    graph.add_node(
        "retrieve_context",
        retrieve_context,
    )

    graph.add_node(
    "summarize_retrieved_context",
    summarize_retrieved_context,
    )

    graph.add_node(
    "normalize_retrieved_context",
    normalize_retrieved_context,
    )

    graph.add_node("analyze_evidence",
                   analyze_evidence_node
    )

    graph.add_node("generate_answer_draft", 
                     make_generate_answer_node(model_name)
    )

    graph.add_node("judge_answer", 
                   make_judge_answer_node("gpt-4.1")
    )

    
    graph.add_edge(START, "interpret_user_request")
    graph.add_edge("interpret_user_request", "load_memory_context")

    graph.add_conditional_edges(
        "load_memory_context",
        route_after_interpretation,
        {
            "retrieve_context": "retrieve_context",
            "generate_answer_draft": "generate_answer_draft",
        },
    )

    graph.add_edge("retrieve_context", "normalize_retrieved_context")
    graph.add_edge("normalize_retrieved_context", "summarize_retrieved_context")
    graph.add_edge("summarize_retrieved_context", "generate_answer_draft")
    graph.add_edge("generate_answer_draft", "judge_answer")
    graph.add_edge("judge_answer", END)

    return graph.compile()