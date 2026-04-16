from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState
from src.agent.nodes import (
    parse_user_query,
    classify_user_intent,
    detect_missing_information,
    extract_product_features,
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
        2. Detect the user's intent
        3. Identify missing information required for feasibility analysis
        4. Normalize product characteristics into ProductFeatures
        5. Route execution to the appropriate response generator:
            - constraint summary
            - partial response (missing information)
            - decision workflow (stub for now)

    The graph operates on a shared state object (AgentState),
    progressively enriched by each node.
    """

    # Create a state-based workflow builder.
    # AgentState defines the shared memory structure used across all nodes.
    builder = StateGraph(AgentState)

    # =========================
    # CORE ANALYSIS NODES
    # =========================

    # Node 1: Parse the raw user query into a structured ProductRequest object.
    builder.add_node("parse_user_query", parse_user_query)

    # Node 2: Classify the request type
    # (decision / summary / comparison / clarification).
    builder.add_node("classify_user_intent", classify_user_intent)

    # Node 3: Detect whether required product attributes are missing.
    builder.add_node("detect_missing_information", detect_missing_information)

    # Node 4: Convert the parsed request into a normalized ProductFeatures object.
    # This creates a stable interface between parsing logic and the future rule engine.
    builder.add_node("extract_product_features", extract_product_features)

    # =========================
    # RESPONSE GENERATION NODES
    # =========================

    # Generate a high-level insurer constraint summary.
    builder.add_node("generate_summary_answer", generate_summary_answer)

    # Generate a partial response when key inputs are missing.
    builder.add_node("generate_partial_answer", generate_partial_answer)

    # Generate a placeholder feasibility decision response.
    # This node will later be replaced by a real rule-based decision engine.
    builder.add_node("generate_decision_stub", generate_decision_stub)

    # =========================
    # MAIN EXECUTION PIPELINE
    # =========================

    # Define the workflow entry point.
    builder.add_edge(START, "parse_user_query")

    # Step 1 -> Step 2: Parse the query, then classify intent.
    builder.add_edge("parse_user_query", "classify_user_intent")

    # Step 2 -> Step 3: Classify intent, then detect missing information.
    builder.add_edge("classify_user_intent", "detect_missing_information")

    # Step 3 -> Step 4: Once completeness has been evaluated,
    # normalize product characteristics into ProductFeatures.
    builder.add_edge("detect_missing_information", "extract_product_features")

    # =========================
    # CONDITIONAL ROUTING LOGIC
    # =========================

    # Route execution dynamically after feature normalization.
    #
    # route_after_analysis decides which response node should run next,
    # based on:
    # - detected user intent
    # - completeness of the request
    # - analysis state already stored in AgentState
    builder.add_conditional_edges(
        "extract_product_features",
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
    # and then terminates workflow execution.
    builder.add_edge("generate_summary_answer", END)
    builder.add_edge("generate_partial_answer", END)
    builder.add_edge("generate_decision_stub", END)

    # Compile the workflow into an executable LangGraph object.
    return builder.compile()