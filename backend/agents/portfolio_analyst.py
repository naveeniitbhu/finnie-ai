import json
from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm import get_llm
from .state import ConversationState

llm = get_llm(temperature=0.3)

SYSTEM_PROMPT = """You are a portfolio analysis expert with deep knowledge of asset allocation, diversification, risk management, and portfolio construction.

Your role:
- Analyze the user's stated holdings against their risk tolerance and investment goals
- Explain concepts like Sharpe ratio, beta, correlation, and rebalancing clearly
- Identify concentration risk, sector imbalances, or misalignment with stated goals
- Always ground analysis in the user's specific profile and live data when provided
- Be conversational and educational — never prescriptive or directive
- Remind users this is educational information, not personalized financial advice"""


def run_portfolio_analyst(state: ConversationState, prior_outputs: dict) -> str:
    profile = state.get("user_profile", {})
    context_chunks = state.get("retrieved_context", [])
    yfinance_data = state.get("yfinance_data") or {}
    query = state.get("current_query", "")
    messages = state.get("messages", [])[-8:]

    rag_block = "\n\n".join(context_chunks) if context_chunks else "No knowledge base context retrieved."
    history = "\n".join(
        f"{m.type}: {m.content}" for m in messages if hasattr(m, "type")
    )

    live_data_block = ""
    if yfinance_data.get("ticker_data"):
        live_data_block = f"\nLive portfolio data:\n{json.dumps(yfinance_data['ticker_data'], indent=2, default=str)}"

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
            f"{live_data_block}"
            f"{prior_block}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"User question: {query}"
        )),
    ])
    return response.content
