from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, field_validator

from backend.llm import get_llm
from .state import ConversationState


class RouterIntent(BaseModel):
    primary_agent: str
    chain: List[str]
    needs_yfinance: bool
    clarification_needed: bool
    clarification_question: str = ""

    @field_validator("primary_agent")
    @classmethod
    def validate_primary_agent(cls, v: str) -> str:
        valid = {
            "portfolio_analyst", "market_trends", "tax_educator",
            "finance_qa", "goal_planning", "news_synthesizer",
        }
        return v if v in valid else "finance_qa"


AGENT_DESCRIPTIONS = {
    "portfolio_analyst": "Analyzes the user's investment portfolio, asset allocation, diversification, and risk assessment based on their stated holdings and risk profile",
    "market_trends": "Explains macroeconomic trends, sector movements, market indices, monetary policy, and general market conditions",
    "tax_educator": "Educates on tax concepts including capital gains, tax-advantaged accounts (401k, IRA, HSA), tax-loss harvesting, and general tax strategies",
    "finance_qa": "Answers general financial education questions about concepts, definitions, investment principles, and financial terminology",
    "goal_planning": "Assists with financial goal setting, retirement planning, savings rate calculations, milestone planning, and timelines",
    "news_synthesizer": "Summarizes and contextualizes recent financial news, earnings reports, and market events and their implications",
}

ROUTER_SYSTEM_PROMPT = """You are a financial assistant router that classifies user queries and routes them to the right specialist agent.

Available agents:
{agent_descriptions}

Analyze the user's message and recent conversation context, then classify the intent:
- primary_agent: the single best-fit agent key from the list above
- chain: ordered list of agent keys to invoke (usually just [primary_agent]; add a second agent only when the query clearly spans two domains, e.g. news + market analysis)
- needs_yfinance: true if the response would benefit from live stock prices, market data, or financial news feeds
- clarification_needed: true only if the query is genuinely too ambiguous to route
- clarification_question: the clarifying question to ask (only when clarification_needed is true)"""

router_llm = get_llm(temperature=0).with_structured_output(RouterIntent)


def router_node(state: ConversationState) -> dict:
    agent_desc_text = "\n".join(f"  {k}: {v}" for k, v in AGENT_DESCRIPTIONS.items())

    recent = state.get("messages", [])[-6:]
    history = "\n".join(
        f"{m.type}: {m.content[:200]}" for m in recent if hasattr(m, "type")
    )

    profile = state.get("user_profile", {})
    profile_summary = (
        f"risk={profile.get('risk_tolerance', 'unknown')}, "
        f"goal={profile.get('primary_goal', 'unknown')}, "
        f"horizon={profile.get('investment_horizon', 'unknown')}, "
        f"tickers={profile.get('tickers', [])}"
    )

    try:
        result: RouterIntent = router_llm.invoke([
            SystemMessage(content=ROUTER_SYSTEM_PROMPT.format(agent_descriptions=agent_desc_text)),
            HumanMessage(content=(
                f"User profile: {profile_summary}\n\n"
                f"Recent conversation:\n{history}\n\n"
                f"Current query: {state.get('current_query', '')}"
            )),
        ])
        intent = result.model_dump()
        if not intent.get("chain"):
            intent["chain"] = [intent["primary_agent"]]
    except Exception:
        intent = {
            "primary_agent": "finance_qa",
            "chain": ["finance_qa"],
            "needs_yfinance": False,
            "clarification_needed": False,
            "clarification_question": "",
        }

    return {"classified_intent": intent}
