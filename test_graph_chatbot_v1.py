import json
from pathlib import Path
from pprint import pprint

from src.agent.graph_chatbot_v1 import build_chatbot_graph


MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",

]

INPUT_PATH = Path("inputs/prompts_test_V1.json")
OUTPUT_DIR = Path("outputs")


def load_prompts(path: Path):
    """
    Load prompt dataset from JSON file.
    """

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_model_tests(model_name, prompts):
    """
    Execute all prompts for a given model.
    """

    print("=" * 80)
    print(f"Testing model: {model_name}")
    print("=" * 80)

    graph = build_chatbot_graph(model_name=model_name)

    results = []

    for item in prompts:
        prompt_id = item["id"]
        category = item["category"]
        query = item["query"]

        print("-" * 60)
        print(f"Prompt {prompt_id} | Category: {category}")
        print(query)

        result = graph.invoke({"user_query": query})

        pprint(result)
        print()

        results.append(
            {
                "id": prompt_id,
                "category": category,
                "model": model_name,
                "query": query,
                "answer": result.get("answer"),
                "error": result.get("error"),
            }
        )

    return results


def save_results(model_name, results):
    """
    Save model results to JSON file.
    """

    OUTPUT_DIR.mkdir(exist_ok=True)

    filename = f"results_{model_name.replace('.', '_')}.json"

    output_path = OUTPUT_DIR / filename

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {output_path}")


def main():
    """
    Run evaluation across all models.
    """

    prompts = load_prompts(INPUT_PATH)

    print(f"Loaded {len(prompts)} prompts")

    for model_name in MODELS:

        results = run_model_tests(model_name, prompts)

        save_results(model_name, results)


if __name__ == "__main__":
    main()