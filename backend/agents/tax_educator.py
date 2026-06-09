import json
from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm import get_llm
from .state import ConversationState

llm = get_llm(temperature=0.2)

SYSTEM_PROMPT = """You are a tax education specialist focused on helping investors understand the tax implications of their financial decisions.

Your role:
- Explain capital gains (short-term vs long-term) and how holding periods affect tax rates
- Clarify tax-advantaged accounts: 401(k), Roth IRA, Traditional IRA, HSA, 529
- Educate on tax-loss harvesting, wash-sale rules, and contribution limits
- Help users understand how their tax filing status and income bracket interact with investment taxes
- Always present current contribution limits and tax brackets when relevant
- Be clear that this is tax education, not tax advice — recommend consulting a CPA for personal situations
- Never recommend specific tax strategies as advice; explain concepts and tradeoffs instead"""


def run_tax_educator(state: ConversationState, prior_outputs: dict) -> str:
    profile = state.get("user_profile", {})
    context_chunks = state.get("retrieved_context", [])
    query = state.get("current_query", "")
    messages = state.get("messages", [])[-8:]

    rag_block = "\n\n".join(context_chunks) if context_chunks else "No knowledge base context retrieved."
    history = "\n".join(
        f"{m.type}: {m.content}" for m in messages if hasattr(m, "type")
    )

    prior_block = ""
    if prior_outputs:
        prior_block = "\nContext from earlier analysis:\n" + "\n".join(
            f"[{k}]: {v}" for k, v in prior_outputs.items()
        )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"User profile: {json.dumps(profile)}\n\n"
            f"Knowledge base:\n{rag_block}"
            f"{prior_block}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"User question: {query}"
        )),
    ])
    return response.content
