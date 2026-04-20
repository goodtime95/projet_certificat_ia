import json
import os

from openai import OpenAI
from pydantic import ValidationError

from src.agent.schemas_chatbot import StructuredChatbotOutput
from src.agent.state_chatbot import ChatbotState


SYSTEM_PROMPT = """
You are an assistant specialized in structured-product referencing for French Retail clients.

Retail structured products are generally distributed inside an insurance wrapper:
French or Luxembourg life insurance contracts.
Each insurer has its own referencing constraints and internal policies.

Your role is to analyze user requests related to the referencing of structured products in these insurance wrappers.

Important rules:
- Do not invent insurer policies, internal rules, or product eligibility criteria.
- If the user's question requires insurer-specific policy information that is not provided, say so clearly.
- Do not present assumptions as confirmed facts.
- If the request is outside the scope of structured-product referencing, mark it as out of scope.
- If the request is too vague, identify exactly which information is missing.
- If the request contains factually incorrect, contradictory, or misleading assumptions, detect and report them.
- Do not claim certainty unless explicit policy evidence is available.
- Be clear, concise, and operational.

You must return a structured JSON object matching the target schema.
Do not return free text outside the JSON object.

Interpret the user's request into:
1. intent
2. parsed_request
3. missing_fields
4. detected_inconsistencies
5. scope_status
6. final_answer

Allowed values:

- intent:
  - referencing_feasibility
  - referencing_comparison
  - policy_confirmation
  - out_of_scope
  - general_knowledge
  - unclear

- scope_status:
  - in_scope
  - out_of_scope
  - needs_clarification

- final_answer.mode:
  - clarify
  - reject_incorrect_premise
  - preliminary_decision
  - out_of_scope

Priority rules:
1. If the request is outside the scope of referencing analysis, use:
   - intent = out_of_scope
   - scope_status = out_of_scope
   - final_answer.mode = out_of_scope
2. If the request contains a clearly incorrect, contradictory, or misleading premise, use:
   - final_answer.mode = reject_incorrect_premise
   even if some information is also missing.
3. If the request is in scope but lacks key information, use:
   - scope_status = needs_clarification
   - final_answer.mode = clarify
4. Use preliminary_decision only when the request is in scope and sufficiently specified for a bounded first-pass assessment.

Field guidelines:
- Keep fields null or empty when information is missing.
- List missing information explicitly in missing_fields.
- Use only stable missing_fields labels when relevant:
  - insurer
  - contract
  - issuer
  - product_structure
  - product_type
  - payoff_type
  - underlying
  - maturity
  - platform
  - official_policy_source
- Use features for secondary attributes that do not fit cleanly into product_type, payoff_type, or underlyings.
  Examples: PDI, KI 60, memory coupon, capital guarantee.
- Do not duplicate the same information across multiple fields unless necessary.

Multi-product requests:
- If the user mentions multiple products, preserve the comparative nature of the request.
- Do not invent missing detail to force a single-product interpretation.
- Capture the main comparative intent at top level, and preserve secondary product detail as faithfully as possible within parsed_request and features.

In detected_inconsistencies, use short stable codes when relevant, such as:
- UC_VS_FONDS_EURO
- EMTN_VS_UCITS_CONFUSION
- UNVERIFIED_POLICY_ASSERTION
- PRODUCT_WRAPPER_MISMATCH
- INSUFFICIENT_INFORMATION
""".strip()

OUTPUT_SCHEMA_DESCRIPTION = """
Return JSON only, using exactly this structure:

{
  "intent": "referencing_feasibility",
  "parsed_request": {
    "insurers": [],
    "product_type": null,
    "payoff_type": null,
    "underlying_type": null,
    "underlyings": [],
    "maturity": null,
    "issuer": null,
    "wrapper": null,
    "features": []
  },
  "missing_fields": [],
  "detected_inconsistencies": [
    {
      "code": "STRING_CODE",
      "message": "Short explanation"
    }
  ],
  "scope_status": "in_scope",
  "final_answer": {
    "mode": "clarify",
    "summary": "Short operational summary"
  }
}
""".strip()


def call_llm_structured(user_query: str, model_name: str) -> StructuredChatbotOutput:
    """
    Call the LLM and validate the returned JSON against the expected schema.
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
                "role": "system",
                "content": OUTPUT_SCHEMA_DESCRIPTION,
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
    )

    raw_text = response.output_text.strip()

    try:
        parsed_json = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Model did not return valid JSON.\n"
            f"Raw output:\n{raw_text}"
        ) from exc

    try:
        return StructuredChatbotOutput.model_validate(parsed_json)
    except ValidationError as exc:
        raise ValueError(
            "Model returned JSON, but it does not match the expected schema.\n"
            f"Raw JSON:\n{json.dumps(parsed_json, ensure_ascii=False, indent=2)}\n"
            f"Validation error:\n{exc}"
        ) from exc


def make_generate_structured_chatbot_answer(model_name: str):
    """
    Create a LangGraph-compatible node bound to a specific model name.
    """

    def generate_structured_chatbot_answer(state: ChatbotState) -> ChatbotState:
        """
        Generate a structured interpretation of the user request.
        """

        user_query = state["user_query"]

        try:
            structured_output = call_llm_structured(
                user_query=user_query,
                model_name=model_name,
            )

            return {
                "structured_output": structured_output.model_dump(),
                "model_used": model_name,
            }

        except Exception as exc:
            return {
                "error": str(exc),
                "structured_output": {
                    "intent": "unclear",
                    "parsed_request": {
                        "insurers": [],
                        "product_type": None,
                        "payoff_type": None,
                        "underlying_type": None,
                        "underlyings": [],
                        "maturity": None,
                        "issuer": None,
                        "wrapper": None,
                        "features": [],
                    },
                    "missing_fields": [],
                    "detected_inconsistencies": [
                        {
                            "code": "PARSING_ERROR",
                            "message": "An error occurred while generating structured output.",
                        }
                    ],
                    "scope_status": "needs_clarification",
                    "final_answer": {
                        "mode": "clarify",
                        "summary": (
                            "An error occurred while interpreting the request. "
                            "See the error field for details."
                        ),
                    },
                },
                "model_used": model_name,
            }

    return generate_structured_chatbot_answer