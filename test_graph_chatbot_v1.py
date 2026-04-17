import json
from pathlib import Path
from pprint import pprint

from src.agent.graph_chatbot_v1 import build_chatbot_graph


MODEL_NAME = "gpt-4.1"
OUTPUT_PATH = Path("outputs/chatbot_tests_gpt_4_1.json")


def main():
    """
    Run a minimal test of the chatbot graph and save results to JSON.
    """

    graph = build_chatbot_graph(model_name=MODEL_NAME)

    test_queries = [
        "Je veux faire référencer un autocall worst-of chez SwissLife, est-ce possible ?",
        "Quelles sont les contraintes principales de référencement chez AEP ?",
        "Mon client aime les actions, quel produit structuré proposer ?",
    ]

    results = []

    print(f"Model under test: {MODEL_NAME}")
    print()

    for i, query in enumerate(test_queries, start=1):
        print("=" * 80)
        print(f"TEST {i}")
        print(query)

        result = graph.invoke({"user_query": query})
        pprint(result)
        print()

        results.append(
            {
                "test_id": i,
                "model": result.get("model_used", MODEL_NAME),
                "query": query,
                "answer": result.get("answer"),
                "error": result.get("error"),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()