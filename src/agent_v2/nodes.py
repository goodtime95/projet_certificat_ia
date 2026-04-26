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
    SourceNeed,
)
from src.agent_v2.state import AgentV2State

from src.retrieval.context_retriever import retrieve_context_from_index

from pathlib import Path
from typing import Any, Dict, List

REFERENCEMENT_DIR = Path("data/referencement")

INSURER_FILE_MAP = {
    "axa": "axa.md",
    "aep": "aep.md",
    "generali": "generali.md",
}

EMAIL_HISTORY_DIR = Path("data/email_history_mock")

EMAIL_FILE_MAP = {
    "axa": "axa.md",
    "aep": "aep.md",
    "generali": "generali.md",
}


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
                                 f"{state.get('retrieved_context', [])}"),
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



def retrieve_context(state: AgentV2State) -> AgentV2State:
    """
    Retrieve context from the local multi-source vector index.

    The runtime agent does not read raw files directly.
    It only consumes the offline-built vector index from data_agent_vect.
    """

    interpreted = state.get("interpreted_request")

    if interpreted is None:
        return {
            "retrieved_context": [],
            "error": "No interpreted_request available for context retrieval.",
        }

    source_types = [source.value for source in interpreted.required_sources]

    if not source_types:
        return {
            "retrieved_context": [],
        }

    try:
        retrieved_context = retrieve_context_from_index(
            query=state["user_query"],
            insurers=interpreted.insurers,
            source_types=source_types,
            k=6,
        )

        return {
            "retrieved_context": retrieved_context,
        }

    except Exception as exc:
        return {
            "retrieved_context": [],
            "error": str(exc),
        }