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
last_retrieved_context = None
last_answer = None


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
        last_retrieved_context = None
        last_answer = None

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
                "last_retrieved_context": last_retrieved_context,
                "last_answer": last_answer,
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

    last_retrieved_context = result.get(
    "last_retrieved_context",
    result.get("retrieved_context", last_retrieved_context),
        )

    last_answer = result.get(
    "last_answer",
    answer,
        )
    
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

    if answer.raw_sources:
        print("\nraw_sources :")

        for source in answer.raw_sources:
            print("\n" + "-" * 60)
            print(f"source_id   : {source.source_id}")
            print(f"entity      : {source.entity}")
            print(f"source_type : {source.source_type}")
            print(f"source_name : {source.source_name}")
            print(f"page        : {source.page}")

            if source.excerpt:
                print("\nexcerpt :")
                print(source.excerpt)

            if source.raw_text:
                print("\nraw_text :")
                print(source.raw_text)

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
            "last_retrieved_context": last_retrieved_context,
            "retrieval_status": result.get("retrieval_status"),
            "answer": answer.model_dump(),
            "last_answer": (
                last_answer.model_dump()
                if last_answer
                else None
            ),
            "model_used": result.get("model_used"),
            "error": result.get("error"),
        }
    )