from langchain_openai import ChatOpenAI
from src.agent_v2.prompts import (
    INTERPRETATION_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
)
from src.agent_v2.schemas import (
    InterpretedRequest,
    AgentAnswer,
    Intent,
    ResponseMode,
)
from src.agent_v2.state import AgentV2State

from pathlib import Path
from typing import Any, Dict, List


def make_interpret_user_request_node(model_name: str):

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    structured_llm = llm.with_structured_output(InterpretedRequest)

    def interpret_user_request(state: AgentV2State):

        query = state["user_query"]

        try:

            result = structured_llm.invoke(
                [
                    {"role": "system", "content": INTERPRETATION_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ]
            )

            return {
                "interpreted_request": result,
                "model_used": model_name,
            }

        except Exception as e:

            return {
                "error": str(e)
            }

    return interpret_user_request


def make_generate_answer_node(model_name: str):

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    structured_llm = llm.with_structured_output(AgentAnswer)

    def generate_answer(state: AgentV2State):

        interpreted = state["interpreted_request"]
        
        if interpreted.intent == Intent.PRODUCT_ADVICE:
            return {
                "answer" : AgentAnswer(
                mode=ResponseMode.CLARIFY,
                summary=(
                    "I cannot recommend structured products to sell. "
                    "I can help assess referencing feasibility once insurer, wrapper, "
                    "issuer, maturity, payoff structure and underlyings are specified."),
                next_steps=[
                    "Specify target insurer or platform",
                    "Specify wrapper",
                    "Provide issuer",
                    "Provide maturity",
                    "Provide payoff structure",
                    "Provide underlyings"
                    ],
                    warnings=[],)
                    }

        try:

            result = structured_llm.invoke(
                [
                    {"role": "user",
                     "content": ("INTERPRETED REQUEST:\n"
                                 f"{interpreted.model_dump_json(indent=2)}\n\n"
                                 "RETRIEVED POLICY CHUNKS:\n"
                                 f"{state.get('retrieved_chunks', [])}"),
                    },
                ]
            )

            return {
                "answer": result
            }

        except Exception as e:

            return {
                "error": str(e)
            }

    return generate_answer


REFERENCEMENT_DIR = Path("data/referencement_mock")

INSURER_FILE_MAP = {
    "axa": "axa.md",
    "aep": "aep.md",
    "generali": "generali.md",
}

def retrieve_policy_chunks(state: AgentV2State) -> AgentV2State:
    """
    Retrieve local referencing policy documents based on detected insurers.

    This is a minimal local RAG-like retrieval layer.
    It does not use embeddings yet.

    The goal is to connect:
    interpreted_request.insurers -> retrieved_chunks

    Each retrieved chunk contains:
    - insurer
    - source path
    - content
    """

    interpreted = state.get("interpreted_request")

    if interpreted is None:
        return {
            "retrieved_chunks": [],
            "error": "No interpreted_request available for retrieval.",
        }

    retrieved_chunks: List[Dict[str, Any]] = []

    for insurer in interpreted.insurers:
        insurer_key = insurer.lower().strip()
        filename = INSURER_FILE_MAP.get(insurer_key)

        if filename is None:
            continue

        file_path = REFERENCEMENT_DIR / filename

        if not file_path.exists():
            retrieved_chunks.append(
                {
                    "insurer": insurer,
                    "source": str(file_path),
                    "content": "",
                    "error": "Policy file not found.",
                }
            )
            continue

        content = file_path.read_text(encoding="utf-8")

        retrieved_chunks.append(
            {
                "insurer": insurer,
                "source": str(file_path),
                "content": content,
            }
        )

    return {
        "retrieved_chunks": retrieved_chunks,
    }