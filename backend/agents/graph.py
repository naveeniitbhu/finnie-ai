from langgraph.graph import StateGraph, END

from .state import ConversationState
from .router import router_node
from .portfolio_analyst import run_portfolio_analyst
from .market_trends import run_market_trends
from .tax_educator import run_tax_educator
from .finance_qa import run_finance_qa
from .goal_planning import run_goal_planning
from .news_synthesizer import run_news_synthesizer
from backend.rag.retriever import rag_node
from backend.tools.yfinance_tool import yfinance_node

AGENT_FUNCTIONS = {
    "portfolio_analyst": run_portfolio_analyst,
    "market_trends": run_market_trends,
    "tax_educator": run_tax_educator,
    "finance_qa": run_finance_qa,
    "goal_planning": run_goal_planning,
    "news_synthesizer": run_news_synthesizer,
}

AGENT_LABELS = {
    "portfolio_analyst": "Portfolio Analyst",
    "market_trends": "Market Trends Analyst",
    "tax_educator": "Tax Educator",
    "finance_qa": "Finance Q&A",
    "goal_planning": "Goal Planning Advisor",
    "news_synthesizer": "News Synthesizer",
}


def clarification_node(state: ConversationState) -> dict:
    intent = state.get("classified_intent") or {}
    question = intent.get("clarification_question", "Could you clarify your question? I want to make sure I give you the most relevant answer.")
    return {"final_response": question, "agent_used": "Router"}


def agent_executor_node(state: ConversationState) -> dict:
    intent = state.get("classified_intent") or {}
    chain = intent.get("chain") or [intent.get("primary_agent", "finance_qa")]

    # Filter to only valid agent names
    chain = [a for a in chain if a in AGENT_FUNCTIONS]
    if not chain:
        chain = ["finance_qa"]

    prior_outputs: dict = {}
    for agent_name in chain:
        output = AGENT_FUNCTIONS[agent_name](state, prior_outputs)
        prior_outputs[agent_name] = output

    last_agent = chain[-1]
    final_response = prior_outputs.get(last_agent, "I wasn't able to generate a response. Please try again.")
    agent_label = AGENT_LABELS.get(last_agent, last_agent)

    return {
        "agent_outputs": prior_outputs,
        "final_response": final_response,
        "agent_used": agent_label,
    }


def route_after_rag(state: ConversationState) -> str:
    intent = state.get("classified_intent") or {}
    if intent.get("clarification_needed"):
        return "clarification"
    if intent.get("needs_yfinance"):
        return "yfinance"
    return "agent_executor"


def build_graph() -> StateGraph:
    builder = StateGraph(ConversationState)

    builder.add_node("router", router_node)
    builder.add_node("rag", rag_node)
    builder.add_node("yfinance", yfinance_node)
    builder.add_node("agent_executor", agent_executor_node)
    builder.add_node("clarification", clarification_node)

    builder.set_entry_point("router")
    builder.add_edge("router", "rag")
    builder.add_conditional_edges(
        "rag",
        route_after_rag,
        {
            "clarification": "clarification",
            "yfinance": "yfinance",
            "agent_executor": "agent_executor",
        },
    )
    builder.add_edge("yfinance", "agent_executor")
    builder.add_edge("agent_executor", END)
    builder.add_edge("clarification", END)

    return builder.compile()


graph = build_graph()
