

from pprint import pprint

from src.agent.graph_v1 import build_graph


TEST_QUERIES = [
    "Je veux faire référencer un autocall worst-of dans un contrat SwissLife, est-ce que c’est possible ?",
    "J’ai un client qui souhaite faire référencer un produit chez CNP, Axa et Generali, est-ce que je peux faire un autocall 10 ans Goldman Sachs sur l’indice Electrification ?",
    "Quelles sont les contraintes principales de référencement chez AEP ?",
    "J'ai un client qui aime bien les actions, quel produit je dois lui proposer ?",
]


def main():
    graph = build_graph()

    for i, query in enumerate(TEST_QUERIES, start=1):
        print("=" * 80)
        print(f"TEST {i}")
        print(query)
        result = graph.invoke({"user_query": query})
        pprint(result)
        print()


if __name__ == "__main__":
    main()