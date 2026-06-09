import json
from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm import get_llm
from .state import ConversationState

llm = get_llm(temperature=0.4)

SYSTEM_PROMPT = """You are a financial news analyst who summarizes and contextualizes financial news for retail investors.

Your role:
- Synthesize recent news headlines and summaries into a coherent narrative
- Explain what the news means for different types of investors (conservative, moderate, aggressive)
- Connect news events to broader market trends, economic cycles, and sector dynamics
- Highlight what is signal vs. noise — not every headline moves markets meaningfully
- When live market data is provided, tie news sentiment to actual price movements
- Be objective: present multiple interpretations when the news is ambiguous
- Keep a clear "so what does this mean for you?" framing based on the user's profile
- Remind users that short-term news often has limited impact on long-term investors"""


def run_news_synthesizer(state: ConversationState, prior_outputs: dict) -> str:
    profile = state.get("user_profile", {})
    context_chunks = state.get("retrieved_context", [])
    yfinance_data = state.get("yfinance_data") or {}
    query = state.get("current_query", "")
    messages = state.get("messages", [])[-8:]

    rag_block = "\n\n".join(context_chunks) if context_chunks else "No knowledge base context retrieved."
    history = "\n".join(
        f"{m.type}: {m.content}" for m in messages if hasattr(m, "type")
    )

    news_block = ""
    if yfinance_data.get("news"):
        news_block = f"\nRecent news:\n{json.dumps(yfinance_data['news'], indent=2)}"
    if yfinance_data.get("market_overview"):
        news_block += f"\nMarket overview:\n{json.dumps(yfinance_data['market_overview'], indent=2)}"

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
            f"{news_block}"
            f"{prior_block}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"User question: {query}"
        )),
    ])
    return response.content
