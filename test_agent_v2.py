# test_agent_v2.py

from pprint import pprint
from src.agent_v2.graph import build_agent_v2_graph


MODEL_NAME = "gpt-4.1-mini"


TEST_QUERIES = [
    # "J’ai deux sujets à pousser : un phoenix worst-of 10 ans sur Generali / BNP et un Athena EuroStoxx 50 6 ans, est-ce que ça se référence plutôt chez AXA ou chez AEP ?",
    # "Je veux lancer un structuré assez standard chez AXA, tu penses qu’on peut le faire référencer vite ?",
    # "On est d’accord qu’AXA prend maintenant automatiquement tous les EMTN 12 ans bien notés, donc je peux déjà partir sur la structure sans même attendre leur retour ?",
    # "Je veux mettre un autocall SX5E en UC dans le fonds euro Generali, c’est bien le bon setup pour l’assurance vie française non ?",
    "Tu me recommanderais quoi comme structuré simple à vendre vite en assurance vie en ce moment ?",
]


def main():
    """
    Run a basic end-to-end test of Agent V2.

    The test checks:
    - structured interpretation
    - multi-product parsing
    - missing information detection
    - incorrect premise detection
    - final operational answer generation
    """

    graph = build_agent_v2_graph(model_name=MODEL_NAME)

    print("=" * 80)
    print(f"Testing Agent V2 with model: {MODEL_NAME}")
    print("=" * 80)

    for idx, query in enumerate(TEST_QUERIES, start=1):
        print("\n" + "=" * 80)
        print(f"TEST {idx}")
        print(query)
        print("-" * 80)

        result = graph.invoke(
            {
                "user_query": query,
            }
        )

        pprint(result)


if __name__ == "__main__":
    main()