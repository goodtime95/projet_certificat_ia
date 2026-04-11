from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState
from src.agent.nodes import (
    parse_user_query,
    classify_user_intent,
    detect_missing_information,
    route_after_analysis,
    generate_summary_answer,
    generate_partial_answer,
    generate_decision_stub,
)


def build_graph():
    """
    Build and compile the LangGraph workflow used by the referencing assistant.

    This workflow defines how the agent processes a user request step by step:

        1. Parse the natural-language query
        2. Detect the user’s intent
        3. Identify missing information required for feasibility analysis
        4. Route execution to the appropriate response generator:
            - constraint summary
            - partial response (missing info)
            - decision workflow (stub for now)

    The graph operates on a shared state object (AgentState),
    progressively enriched by each node.
    """

    # Create a LangGraph state-based workflow builder
    # AgentState defines the shared memory structure across all nodes
    builder = StateGraph(AgentState)


    # =========================
    # CORE ANALYSIS NODES
    # =========================

    # Node 1: Extract structured product information from user query
    builder.add_node("parse_user_query", parse_user_query)

    # Node 2: Classify the request type (decision / summary / comparison / clarification)
    builder.add_node("classify_user_intent", classify_user_intent)

    # Node 3: Detect whether required product attributes are missing
    builder.add_node("detect_missing_information", detect_missing_information)


    # =========================
    # RESPONSE GENERATION NODES
    # =========================

    # Generates a high-level insurer constraint summary
    builder.add_node("generate_summary_answer", generate_summary_answer)

    # Generates a partial response when key inputs are missing
    builder.add_node("generate_partial_answer", generate_partial_answer)

    # Generates a placeholder feasibility decision response
    # (to be replaced later by rule-based decision engine output)
    builder.add_node("generate_decision_stub", generate_decision_stub)


    # =========================
    # MAIN EXECUTION PIPELINE
    # =========================

    # Define workflow entry point
    builder.add_edge(START, "parse_user_query")

    # Parse → classify intent
    builder.add_edge("parse_user_query", "classify_user_intent")

    # Intent classification → missing information detection
    builder.add_edge("classify_user_intent", "detect_missing_information")


    # =========================
    # CONDITIONAL ROUTING LOGIC
    # =========================

    # Dynamically route execution depending on:
    # - detected intent
    # - completeness of product description
    #
    # route_after_analysis returns the name of the next node
    builder.add_conditional_edges(
        "detect_missing_information",
        route_after_analysis,
        {
            "generate_summary_answer": "generate_summary_answer",
            "generate_partial_answer": "generate_partial_answer",
            "generate_decision_stub": "generate_decision_stub",
        },
    )


    # =========================
    # TERMINAL STATES
    # =========================

    # Each response generator produces a FinalAnswer object
    # and terminates the workflow execution

    builder.add_edge("generate_summary_answer", END)
    builder.add_edge("generate_partial_answer", END)
    builder.add_edge("generate_decision_stub", END)


    # Compile the workflow into an executable LangGraph object
    return builder.compile()