import json
from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm import get_llm
from .state import ConversationState

llm = get_llm(temperature=0.3)

SYSTEM_PROMPT = """You are a financial education specialist who makes personal finance and investing concepts accessible to everyone.

Your role:
- Answer questions about financial concepts clearly and without unnecessary jargon
- When jargon is unavoidable, define it immediately after using it
- Use concrete examples and analogies to explain abstract concepts
- Cover topics like: compound interest, dollar-cost averaging, index funds vs active funds, bonds, risk vs return, portfolio theory, budgeting frameworks, emergency funds
- Tailor depth of explanation to what the user seems to know — don't over-explain to experts or under-explain to beginners
- Draw on verified knowledge base content when available
- Keep responses conversational and encourage follow-up questions"""


def run_finance_qa(state: ConversationState, prior_outputs: dict) -> str:
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
