from __future__ import annotations

import os
from typing import TYPE_CHECKING, List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

if TYPE_CHECKING:
    from backend.agents.state import ConversationState

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstore = Chroma(
            collection_name="financial_knowledge",
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        if _vectorstore._collection.count() == 0:
            _seed_knowledge_base(_vectorstore)
    return _vectorstore


def _seed_knowledge_base(vectorstore: Chroma) -> None:
    docs = [
        Document(
            page_content="Asset allocation divides investments among asset categories (stocks, bonds, cash, real estate). A common starting point is subtracting your age from 110 to find the percentage to hold in stocks, with the remainder in bonds.",
            metadata={"domain": "portfolio_analyst", "topic": "asset_allocation"},
        ),
        Document(
            page_content="Diversification spreads investments across assets to reduce unsystematic risk. A well-diversified portfolio typically includes domestic stocks, international stocks, bonds, and alternative assets. Correlation between holdings determines diversification effectiveness.",
            metadata={"domain": "portfolio_analyst", "topic": "diversification"},
        ),
        Document(
            page_content="Rebalancing realigns portfolio weights to the target allocation. Most advisors recommend annual or threshold-based rebalancing (e.g., rebalance when any asset class drifts more than 5% from target). Tax implications of selling should be considered.",
            metadata={"domain": "portfolio_analyst", "topic": "rebalancing"},
        ),
        Document(
            page_content="Beta measures a stock's volatility relative to the market. A beta of 1.0 moves with the market. Beta > 1 amplifies market swings; beta < 1 dampens them. High-beta stocks suit aggressive investors; low-beta suits conservative ones.",
            metadata={"domain": "portfolio_analyst", "topic": "risk_metrics"},
        ),
        Document(
            page_content="The S&P 500 tracks 500 large-cap US companies and is the standard benchmark for US equity performance. Historically it has returned approximately 10% annually before inflation, or roughly 7% in real (inflation-adjusted) terms.",
            metadata={"domain": "market_trends", "topic": "indices"},
        ),
        Document(
            page_content="Inflation erodes purchasing power over time. The Federal Reserve targets 2% annual inflation. When inflation rises above target, the Fed typically raises interest rates, which increases borrowing costs and often slows economic growth and stock valuations.",
            metadata={"domain": "market_trends", "topic": "inflation"},
        ),
        Document(
            page_content="Sector rotation is the practice of moving money between industry sectors as the economy cycles. Defensive sectors (utilities, consumer staples, healthcare) tend to outperform during recessions; cyclical sectors (technology, financials, energy) during expansions.",
            metadata={"domain": "market_trends", "topic": "sector_rotation"},
        ),
        Document(
            page_content="The yield curve plots interest rates across different bond maturities. An inverted yield curve (short-term rates > long-term rates) has historically preceded recessions by 6–18 months and is widely watched as a leading economic indicator.",
            metadata={"domain": "market_trends", "topic": "yield_curve"},
        ),
        Document(
            page_content="Capital gains tax applies to investment profits. Short-term gains (assets held < 1 year) are taxed as ordinary income. Long-term gains (assets held > 1 year) are taxed at 0%, 15%, or 20% depending on taxable income. In 2024, the 15% rate applies for most middle-income earners.",
            metadata={"domain": "tax_educator", "topic": "capital_gains"},
        ),
        Document(
            page_content="A 401(k) is an employer-sponsored plan funded with pre-tax dollars, reducing current taxable income. The 2024 contribution limit is $23,000 ($30,500 for those 50+). Many employers match contributions — always contribute enough to capture the full match, as it is essentially free money.",
            metadata={"domain": "tax_educator", "topic": "401k"},
        ),
        Document(
            page_content="A Roth IRA accepts after-tax contributions and grows tax-free; qualified withdrawals in retirement are also tax-free. 2024 contribution limit: $7,000 ($8,000 if 50+). Income phase-out for single filers: $146,000–$161,000. No required minimum distributions during the owner's lifetime.",
            metadata={"domain": "tax_educator", "topic": "roth_ira"},
        ),
        Document(
            page_content="Tax-loss harvesting sells investments at a loss to offset capital gains, reducing your tax bill. Wash-sale rules prevent buying a substantially identical security within 30 days before or after the sale. This strategy is most valuable in taxable accounts.",
            metadata={"domain": "tax_educator", "topic": "tax_loss_harvesting"},
        ),
        Document(
            page_content="A Health Savings Account (HSA) offers triple tax advantage: contributions are tax-deductible, growth is tax-free, and withdrawals for qualified medical expenses are tax-free. 2024 limits: $4,150 for individuals, $8,300 for families. Requires enrollment in a high-deductible health plan.",
            metadata={"domain": "tax_educator", "topic": "hsa"},
        ),
        Document(
            page_content="Compound interest is interest earned on both the original principal and accumulated interest. The Rule of 72 estimates time to double money: divide 72 by the annual return rate. At 7% annual return, money doubles in approximately 10 years.",
            metadata={"domain": "finance_qa", "topic": "compound_interest"},
        ),
        Document(
            page_content="Dollar-cost averaging (DCA) invests a fixed amount at regular intervals regardless of market conditions. This reduces the emotional impact of volatility, automatically buys more shares when prices are low, and fewer when prices are high. It works best for long-term investors.",
            metadata={"domain": "finance_qa", "topic": "dca"},
        ),
        Document(
            page_content="Index funds track a market index (e.g., S&P 500) passively. They have low expense ratios (often 0.03–0.20%) compared to active funds (often 0.5–1.5%). Decades of research show most active fund managers underperform their benchmark index after fees over long periods.",
            metadata={"domain": "finance_qa", "topic": "index_funds"},
        ),
        Document(
            page_content="Bonds are debt instruments where you lend money to an issuer (government or corporation) in exchange for regular interest payments and return of principal at maturity. Bond prices move inversely to interest rates. Bonds provide income and reduce portfolio volatility.",
            metadata={"domain": "finance_qa", "topic": "bonds"},
        ),
        Document(
            page_content="An emergency fund covers 3–6 months of essential living expenses and should be held in a liquid, low-risk account (HYSA or money market). It is the foundation of financial stability — build this before investing, to avoid needing to liquidate investments at a loss during emergencies.",
            metadata={"domain": "goal_planning", "topic": "emergency_fund"},
        ),
        Document(
            page_content="The 4% withdrawal rule suggests retirees can withdraw 4% of their portfolio in the first year of retirement, adjusting annually for inflation, with a high probability of not outliving savings over 30 years. Based on historical US market returns; lower rates (3–3.5%) are safer for longer retirements.",
            metadata={"domain": "goal_planning", "topic": "retirement_withdrawal"},
        ),
        Document(
            page_content="The FIRE (Financial Independence, Retire Early) framework calculates your 'FIRE number' as 25x your annual expenses. To reach FIRE faster, increase savings rate — at a 50% savings rate, financial independence is achievable in roughly 17 years from zero; at 75%, in about 7 years.",
            metadata={"domain": "goal_planning", "topic": "fire"},
        ),
        Document(
            page_content="SMART goals for finance are: Specific (save $50,000 for a house down payment), Measurable (track monthly contributions), Achievable (realistic given income/expenses), Relevant (aligned with life priorities), Time-bound (within 5 years). Vague goals like 'save more money' rarely succeed.",
            metadata={"domain": "goal_planning", "topic": "smart_goals"},
        ),
    ]
    vectorstore.add_documents(docs)


def retrieve_context(query: str, agent_name: str, k: int = 4) -> List[str]:
    try:
        vs = get_vectorstore()
        docs = vs.similarity_search(query, k=k * 2)
        domain_docs = [d for d in docs if d.metadata.get("domain") == agent_name]
        other_docs = [d for d in docs if d.metadata.get("domain") != agent_name]
        ordered = (domain_docs + other_docs)[:k]
        return [d.page_content for d in ordered]
    except Exception:
        return []


def rag_node(state: "ConversationState") -> dict:
    intent = state.get("classified_intent") or {}
    primary_agent = intent.get("primary_agent", "finance_qa")
    query = state.get("current_query", "")
    context = retrieve_context(query, primary_agent)
    return {"retrieved_context": context}
