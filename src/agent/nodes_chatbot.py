import os

from openai import OpenAI

from src.agent.state_chatbot import ChatbotState


SYSTEM_PROMPT = """
You are an assistant specialized in structured-product referencing.

Your role is to help users answer questions related to the referencing
of structured products by insurers or distribution platforms.

Important rules:
- Do not invent insurer policies, internal rules, or product eligibility criteria.
- If the user's question requires insurer-specific policy information that is not provided, say so clearly.
- Distinguish between:
  1. general structured-product knowledge,
  2. hypothetical reasoning,
  3. actual referencing validation.
- When information is missing, explain what is missing.
- Be clear, concise, and operational.
- Do not present assumptions as confirmed facts.
- If the user asks whether a product is referencable, do not claim certainty unless explicit policy evidence is available.

Your goal is not to hallucinate a decision engine.
Your goal is to provide useful, bounded, and transparent assistance.
""".strip()


def call_llm(user_query: str, model_name: str) -> str:
    """
    Call the LLM with a system prompt and a user question.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
    )

    return response.output_text.strip()


def make_generate_chatbot_answer(model_name: str):
    """
    Create a LangGraph-compatible node bound to a specific model name.

    This allows the graph builder to inject the model configuration
    without storing it inside the state.
    """

    def generate_chatbot_answer(state: ChatbotState) -> ChatbotState:
        """
        Generate a response for a structured-product referencing question
        using the configured LLM model.
        """

        user_query = state["user_query"]

        try:
            answer = call_llm(user_query=user_query, model_name=model_name)
            return {
                "answer": answer,
                "model_used": model_name,
            }

        except Exception as exc:
            return {
                "error": str(exc),
                "answer": (
                    "An error occurred while calling the LLM. "
                    "See the error field for details."
                ),
                "model_used": model_name,
            }

    return generate_chatbot_answer