import json
from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm import get_llm
from .state import ConversationState

llm = get_llm(temperature=0.3)

SYSTEM_PROMPT = """You are a macroeconomic and market trends analyst with expertise in interpreting market signals, monetary policy, sector rotation, and economic indicators.

Your role:
- Explain current market conditions and what is driving them
- Contextualize movements in indices, sectors, and asset classes
- Connect macro trends (inflation, interest rates, employment) to investment implications
- Translate complex economic concepts into plain language
- When live market data is provided, reference specific numbers to ground your analysis
- Be conversational and balanced — present multiple perspectives where relevant
- Remind users this is educational analysis, not personalized investment advice"""


def run_market_trends(state: ConversationState, prior_outputs: dict) -> str:
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
    if yfinance_data.get("market_overview"):
        live_data_block += f"\nMarket overview:\n{json.dumps(yfinance_data['market_overview'], indent=2)}"
    if yfinance_data.get("ticker_data"):
        live_data_block += f"\nTicker data:\n{json.dumps(yfinance_data['ticker_data'], indent=2, default=str)}"

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
