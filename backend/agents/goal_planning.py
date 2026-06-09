import json
from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm import get_llm
from .state import ConversationState

llm = get_llm(temperature=0.3)

SYSTEM_PROMPT = """You are a financial goal planning specialist who helps users define, quantify, and work toward their financial goals.

Your role:
- Help users define SMART financial goals (Specific, Measurable, Achievable, Relevant, Time-bound)
- Calculate savings rates and contribution amounts needed to reach goals
- Explain concepts like the 4% withdrawal rule, retirement readiness, and milestone planning
- Use the user's stated risk tolerance, investment horizon, and goals to tailor projections
- When historical return data is provided, use it to ground projections with realistic assumptions
- Be honest about trade-offs: higher returns require higher risk; shorter timelines require higher savings rates
- Cover goal types: emergency fund, home purchase, retirement, education, financial independence
- Remind users these are projections based on assumptions, not guarantees"""


def run_goal_planning(state: ConversationState, prior_outputs: dict) -> str:
    profile = state.get("user_profile", {})
    context_chunks = state.get("retrieved_context", [])
    yfinance_data = state.get("yfinance_data") or {}
    query = state.get("current_query", "")
    messages = state.get("messages", [])[-8:]

    rag_block = "\n\n".join(context_chunks) if context_chunks else "No knowledge base context retrieved."
    history = "\n".join(
        f"{m.type}: {m.content}" for m in messages if hasattr(m, "type")
    )

    historical_block = ""
    if yfinance_data.get("historical_returns"):
        historical_block = f"\nHistorical return data:\n{json.dumps(yfinance_data['historical_returns'], indent=2)}"

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
            f"{historical_block}"
            f"{prior_block}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"User question: {query}"
        )),
    ])
    return response.content
