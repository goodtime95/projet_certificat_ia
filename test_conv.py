from datetime import datetime
from pathlib import Path
import json

from src.agent_v2.graph import build_agent_v2_graph


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "gpt-4.1-mini"

TRACE_DIR = Path("logs")
TRACE_DIR.mkdir(parents=True, exist_ok=True)

TRACE_FILE = TRACE_DIR / "conversation_trace.jsonl"


STOP_KEYWORDS = {
    "stop",
    "quit",
    "exit",
}

RESET_KEYWORDS = {
    "reset",
    "clear",
    "efface",
    "efface l'historique",
    "nouvelle conversation",
    "on repart de zéro",
}


# ============================================================
# TRACE LOGGER
# ============================================================

def append_trace(payload: dict):
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# GRAPH
# ============================================================

graph = build_agent_v2_graph(MODEL_NAME)


# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation_history = []
conversation_context = None


# ============================================================
# MAIN LOOP
# ============================================================

print("=" * 80)
print(f"Conversation Agent V2 - model={MODEL_NAME}")
print("=" * 80)

while True:

    user_query = input("\nUSER > ").strip()

    if not user_query:
        continue

    query_lower = user_query.lower().strip()

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    if query_lower in STOP_KEYWORDS:
        print("\nConversation terminated.")
        break

    # --------------------------------------------------------
    # RESET MEMORY
    # --------------------------------------------------------

    if query_lower in RESET_KEYWORDS:
        conversation_history = []
        conversation_context = None

        print("\nConversation memory cleared.")
        continue

    # --------------------------------------------------------
    # ADD USER MESSAGE TO HISTORY
    # --------------------------------------------------------

    conversation_history.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    # --------------------------------------------------------
    # INVOKE GRAPH
    # --------------------------------------------------------

    try:
        result = graph.invoke(
            {
                "user_query": user_query,
                "conversation_history": conversation_history[:-1],
                "conversation_context": conversation_context,
            }
        )

    except Exception as e:
        print("\nASSISTANT >")
        print("Erreur pendant l'exécution du graph.")
        print(str(e))

        append_trace(
            {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "conversation_history": conversation_history,
                "conversation_context": (
                    conversation_context.model_dump()
                    if conversation_context
                    else None
                ),
                "error": str(e),
            }
        )

        continue

    # --------------------------------------------------------
    # UPDATE STRUCTURED CONVERSATION CONTEXT
    # --------------------------------------------------------

    conversation_context = result.get(
        "conversation_context",
        conversation_context,
    )

    # --------------------------------------------------------
    # EXTRACT ANSWER
    # --------------------------------------------------------

    answer = result.get("answer")

    if answer is None:
        print("\nASSISTANT >")
        print("Aucune réponse générée.")

        append_trace(
            {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "conversation_history": conversation_history,
                "conversation_context": (
                    conversation_context.model_dump()
                    if conversation_context
                    else None
                ),
                "error": "No answer returned by graph.",
                "raw_result": str(result),
            }
        )

        continue

    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    print("\nASSISTANT >")
    print(answer.answer)

    print("\n---")
    print(f"mode       : {answer.mode}")
    print(f"confidence : {answer.confidence}")

    if answer.missing_information:
        print("\nmissing_information :")
        for item in answer.missing_information:
            print(f"- {item}")

    if answer.sources_used:
        print("\nsources_used :")
        for source in answer.sources_used:
            print(
                f"- {source.entity} | "
                f"{source.source_type} | "
                f"{source.source_name} | "
                f"page={source.page}"
            )

    # --------------------------------------------------------
    # ADD ASSISTANT MESSAGE TO HISTORY
    # --------------------------------------------------------

    conversation_history.append(
        {
            "role": "assistant",
            "content": answer.answer,
        }
    )

    # --------------------------------------------------------
    # TRACE LOGGING
    # --------------------------------------------------------

    interpreted_request = result.get("interpreted_request")

    append_trace(
        {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "conversation_history": conversation_history,
            "conversation_context": (
                conversation_context.model_dump()
                if conversation_context
                else None
            ),
            "interpreted_request": (
                interpreted_request.model_dump()
                if interpreted_request
                else None
            ),
            "retrieved_context": result.get("retrieved_context"),
            "retrieval_status": result.get("retrieval_status"),
            "answer": answer.model_dump(),
            "model_used": result.get("model_used"),
            "error": result.get("error"),
        }
    )