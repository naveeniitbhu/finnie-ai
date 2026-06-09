from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import TYPE_CHECKING, Any, Callable, Dict, List, TypeVar

import yfinance as yf

if TYPE_CHECKING:
    from backend.agents.state import ConversationState

T = TypeVar("T")
_YFINANCE_TIMEOUT = 12  # seconds per individual fetch call


def _run_with_timeout(fn: Callable[[], T], default: T, timeout: int = _YFINANCE_TIMEOUT) -> T:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except (FuturesTimeout, Exception):
            return default


def fetch_ticker_data(tickers: List[str]) -> Dict[str, Any]:
    result = {}
    for ticker in tickers:
        def _fetch(t=ticker):
            obj = yf.Ticker(t)
            info = obj.info
            hist = obj.history(period="5d")
            return {
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "previous_close": info.get("previousClose"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "5d_change_pct": _pct_change(hist["Close"]) if not hist.empty else None,
            }
        result[ticker] = _run_with_timeout(_fetch, default={"error": "fetch timed out"})
    return result


def fetch_market_overview() -> Dict[str, Any]:
    indices = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^NSEI": "Nifty 50",
        "^VIX": "VIX",
    }
    result = {}
    for symbol, name in indices.items():
        def _fetch(sym=symbol):
            hist = yf.Ticker(sym).history(period="2d")
            if len(hist) >= 2:
                prev = hist["Close"].iloc[-2]
                curr = hist["Close"].iloc[-1]
                return {
                    "price": round(float(curr), 2),
                    "change_pct": round(((curr - prev) / prev) * 100, 2),
                }
            return None
        data = _run_with_timeout(_fetch, default=None)
        if data:
            result[name] = data
    return result


def fetch_ticker_news(tickers: List[str]) -> Dict[str, Any]:
    news_by_ticker: Dict[str, Any] = {}
    for ticker in tickers[:3]:
        def _fetch(t=ticker):
            raw = yf.Ticker(t).news or []
            return [
                {
                    "title": n.get("title"),
                    "summary": n.get("summary", ""),
                    "publisher": n.get("publisher"),
                }
                for n in raw[:5]
            ]
        news_by_ticker[ticker] = _run_with_timeout(_fetch, default=[])
    return news_by_ticker


def fetch_historical_returns(ticker: str, years: int = 10) -> Dict[str, Any]:
    def _fetch():
        hist = yf.Ticker(ticker).history(period=f"{years}y")
        if hist.empty:
            return {}
        start = float(hist["Close"].iloc[0])
        end = float(hist["Close"].iloc[-1])
        total_return = ((end - start) / start) * 100
        annualized = ((end / start) ** (1 / years) - 1) * 100
        return {
            "ticker": ticker,
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(annualized, 2),
            "years": years,
        }
    return _run_with_timeout(_fetch, default={"error": "fetch timed out"})


def _pct_change(series) -> float | None:
    if len(series) < 2:
        return None
    return round(((float(series.iloc[-1]) - float(series.iloc[0])) / float(series.iloc[0])) * 100, 2)


def yfinance_node(state: "ConversationState") -> dict:
    intent = state.get("classified_intent") or {}
    profile = state.get("user_profile") or {}
    tickers: List[str] = profile.get("tickers") or []
    primary_agent = intent.get("primary_agent", "")

    data: Dict[str, Any] = {}

    if primary_agent == "portfolio_analyst" and tickers:
        data["ticker_data"] = fetch_ticker_data(tickers)

    elif primary_agent == "market_trends":
        data["market_overview"] = fetch_market_overview()
        if tickers:
            data["ticker_data"] = fetch_ticker_data(tickers)

    elif primary_agent == "news_synthesizer":
        # Include Nifty and major indices by default for broad market news
        fallback = tickers if tickers else ["^NSEI", "SPY", "QQQ"]
        data["news"] = fetch_ticker_news(fallback)
        data["market_overview"] = fetch_market_overview()

    elif primary_agent == "goal_planning" and tickers:
        data["historical_returns"] = {t: fetch_historical_returns(t) for t in tickers[:2]}

    return {"yfinance_data": data}
