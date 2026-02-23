"""
Portfolio Analytics agent implemented with LangGraph.

Pipeline:
1) load_portfolio
2) build_benchmark_context (sector weights vs S&P 500)
3) analyze_holdings (yfinance fundamentals + technicals + rule scores)
4) compute_risk (portfolio-level risk/concentration metrics)
5) generate_analysis (LLM narrative + optional per-holding LLM signals)
"""
from __future__ import annotations

import json
from typing import Annotated, Sequence, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from utils import load_llm_from_env, load_portfolio as _load_portfolio_file

# Approximate S&P 500 sector weights (static snapshot; used only as a benchmark reference)
SP500_SECTOR_WEIGHTS = {
    "Information Technology": 28.0,
    "Health Care": 13.0,
    "Financials": 12.0,
    "Consumer Discretionary": 10.0,
    "Communication Services": 9.0,
    "Industrials": 8.0,
    "Consumer Staples": 6.0,
    "Energy": 4.0,
    "Utilities": 3.0,
    "Real Estate": 3.0,
    "Materials": 2.0,
}

ANALYSIS_PROMPT = """You are a Portfolio Analytics assistant. You will be given:
- A benchmark sector comparison (portfolio vs S&P 500 weights),
- Per-holding technical/fundamental metrics, rule-based scores & signals,
- Portfolio-level risk metrics.

Write a concise report with:
1) Portfolio Overview (top holdings, diversification notes)
2) Sector & benchmark comparison (over/underweights)
3) Risk summary (beta, concentration, high-risk holdings, data gaps)
4) Actionable observations (educational only; not financial advice)

Avoid overly long output. Use clear bullet points and mention any missing data."""

# ----------------- State -----------------
class AnalyticsState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = []
    portfolio: dict = Field(default_factory=dict)
    benchmark_context: str = ""
    sector_rows: list[dict] = Field(default_factory=list)
    holdings_analysis: list[dict] = Field(default_factory=list)
    risk_metrics: dict = Field(default_factory=dict)


# ----------------- Market data helpers -----------------
def _scalar(x):
    """Convert numpy/pandas scalar to plain Python float."""
    try:
        if hasattr(x, "item"):
            return x.item()
        if pd.isna(x):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV history from yfinance."""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_fundamentals(ticker: str) -> dict:
    """Fetch fundamental metrics from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "company_name": info.get("shortName") or info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "market_cap": info.get("marketCap"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "profit_margins": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
        }
    except Exception:
        return {"company_name": ticker, "sector": "Unknown"}


def compute_technical_indicators(df: pd.DataFrame) -> dict | None:
    """Compute RSI, MACD, SMA from price history."""
    if df.empty or len(df) < 50:
        return None

    close = df["Close"]

    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean() if len(df) >= 200 else pd.Series(dtype=float)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    return {
        "current_price": _scalar(latest["Close"]),
        "sma_50": _scalar(sma_50.iloc[-1]),
        "sma_200": _scalar(sma_200.iloc[-1]) if not sma_200.empty else None,
        "macd": _scalar(macd_line.iloc[-1]),
        "signal_line": _scalar(signal_line.iloc[-1]),
        "rsi": _scalar(rsi.iloc[-1]),
        "high_52w": _scalar(df["High"].max()) if "High" in df.columns else None,
        "low_52w": _scalar(df["Low"].min()) if "Low" in df.columns else None,
    }


def compute_rule_scores(technicals: dict, fundamentals: dict) -> dict:
    """Compute rule-based scores from technical + fundamental data (17 rules)."""
    price = technicals.get("current_price")
    sma50 = technicals.get("sma_50")
    sma200 = technicals.get("sma_200")
    macd = technicals.get("macd")
    signal = technicals.get("signal_line")
    rsi = technicals.get("rsi")
    high52 = technicals.get("high_52w")
    low52 = technicals.get("low_52w")

    pe = fundamentals.get("pe_ratio")
    pb = fundamentals.get("pb_ratio")
    roe = fundamentals.get("roe")
    dividend = fundamentals.get("dividend_yield")
    beta = fundamentals.get("beta")
    profit_margins = fundamentals.get("profit_margins")
    revenue_growth = fundamentals.get("revenue_growth")
    debt_to_equity = fundamentals.get("debt_to_equity")

    score = 0
    rules = {}

    # Technical momentum
    if price and sma50:
        rules["price_above_sma50"] = 1 if price > sma50 else -1
        score += rules["price_above_sma50"]
    if sma50 and sma200:
        rules["sma50_above_sma200"] = 1 if sma50 > sma200 else -1
        score += rules["sma50_above_sma200"]
    if macd is not None and signal is not None:
        rules["macd_above_signal"] = 1 if macd > signal else -1
        score += rules["macd_above_signal"]
    if rsi is not None:
        if rsi < 30:
            rules["rsi_oversold"] = 1
        elif rsi > 70:
            rules["rsi_overbought"] = -1
        else:
            rules["rsi_neutral"] = 0
        score += list(rules.values())[-1]

    # 52w position
    if price and high52 and low52 and high52 != low52:
        pos = (price - low52) / (high52 - low52)
        if pos < 0.2:
            rules["near_52w_low"] = 1
            score += 1
        elif pos > 0.8:
            rules["near_52w_high"] = -1
            score -= 1
        else:
            rules["mid_52w_range"] = 0

    # Valuation
    if pe is not None:
        if pe < 15:
            rules["pe_attractive"] = 1
            score += 1
        elif pe > 30:
            rules["pe_expensive"] = -1
            score -= 1
        else:
            rules["pe_neutral"] = 0
    if pb is not None:
        if pb < 3:
            rules["pb_reasonable"] = 1
            score += 1
        elif pb > 8:
            rules["pb_expensive"] = -1
            score -= 1
        else:
            rules["pb_neutral"] = 0

    # Quality / profitability
    if roe is not None:
        if roe > 0.15:
            rules["roe_strong"] = 1
            score += 1
        elif roe < 0.05:
            rules["roe_weak"] = -1
            score -= 1
        else:
            rules["roe_ok"] = 0
    if profit_margins is not None:
        if profit_margins > 0.15:
            rules["margins_strong"] = 1
            score += 1
        elif profit_margins < 0.05:
            rules["margins_weak"] = -1
            score -= 1
        else:
            rules["margins_ok"] = 0

    # Growth
    if revenue_growth is not None:
        if revenue_growth > 0.10:
            rules["revenue_growth_strong"] = 1
            score += 1
        elif revenue_growth < 0:
            rules["revenue_growth_negative"] = -1
            score -= 1
        else:
            rules["revenue_growth_ok"] = 0

    # Income / dividends
    if dividend is not None:
        if dividend > 0.02:
            rules["dividend_pays"] = 1
            score += 1
        elif dividend == 0:
            rules["no_dividend"] = 0
        else:
            rules["dividend_low"] = 0

    # Risk heuristics
    if beta is not None:
        if beta > 1.5:
            rules["beta_high"] = -1
            score -= 1
        elif beta < 0.8:
            rules["beta_low"] = 1
            score += 1
        else:
            rules["beta_normal"] = 0

    if debt_to_equity is not None:
        if debt_to_equity > 150:
            rules["leverage_high"] = -1
            score -= 1
        elif debt_to_equity < 50:
            rules["leverage_low"] = 1
            score += 1
        else:
            rules["leverage_ok"] = 0

    # Cap extremes
    score = int(max(min(score, 10), -10))
    rules["total_score"] = score
    return rules


def rule_recommendation(total_score: int) -> str:
    if total_score >= 4:
        return "BUY"
    if total_score <= -4:
        return "SELL"
    return "HOLD"


def combine_signals(rule_signal: str, llm_signal: str) -> tuple[str, float]:
    """Combine rule signal (60%) with LLM signal (40%)."""
    to_score = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}
    r = to_score.get(rule_signal.upper(), 0.0)
    l = to_score.get(llm_signal.upper(), 0.0)
    combined = 0.6 * r + 0.4 * l
    if combined > 0.25:
        return "BUY", combined
    if combined < -0.25:
        return "SELL", combined
    return "HOLD", combined


def analyze_holding(ticker: str, allocation_pct: float) -> dict:
    """Fetch data and compute technical + fundamental scores for one holding."""
    fundamentals = fetch_fundamentals(ticker)
    df = fetch_price_history(ticker)

    technicals = compute_technical_indicators(df) if not df.empty else None
    if not technicals:
        return {
            "ticker": ticker.upper(),
            "allocation_pct": float(allocation_pct),
            "company_name": fundamentals.get("company_name", ticker),
            "sector": fundamentals.get("sector", "Unknown"),
            "data_available": False,
            "note": "Insufficient price history for technical indicators.",
        }

    rule_scores = compute_rule_scores(technicals, fundamentals)
    total_score = rule_scores.get("total_score", 0)
    rule_signal = rule_recommendation(total_score)

    out = {
        "ticker": ticker.upper(),
        "allocation_pct": float(allocation_pct),
        "company_name": fundamentals.get("company_name", ticker),
        "sector": fundamentals.get("sector", "Unknown"),
        "data_available": True,
        **technicals,
        **{k: fundamentals.get(k) for k in [
            "pe_ratio", "pb_ratio", "roe", "dividend_yield", "beta", "market_cap",
            "profit_margins", "revenue_growth", "debt_to_equity"
        ]},
        "rule_scores": rule_scores,
        "total_score": total_score,
        "rule_signal": rule_signal,
    }
    return out


def compute_portfolio_risk(holdings_analysis: list[dict]) -> dict:
    """Compute portfolio-level risk metrics from per-holding analysis."""
    weights = np.array([h.get("allocation_pct", 0.0) for h in holdings_analysis], dtype=float)
    weights = weights / weights.sum() if weights.sum() else weights

    betas = np.array([h.get("beta") or np.nan for h in holdings_analysis], dtype=float)
    pes = np.array([h.get("pe_ratio") or np.nan for h in holdings_analysis], dtype=float)

    weighted_avg_beta = float(np.nansum(weights * betas)) if np.isfinite(betas).any() else None
    weighted_avg_pe = float(np.nansum(weights * pes)) if np.isfinite(pes).any() else None

    # Concentration (HHI on percent weights)
    w_pct = np.array([h.get("allocation_pct", 0.0) for h in holdings_analysis], dtype=float)
    w_frac = w_pct / 100.0 if w_pct.sum() else w_pct
    concentration_hhi = float(np.sum(w_frac ** 2)) if w_frac.size else None
    max_single = float(w_pct.max()) if w_pct.size else None

    # High-risk holdings heuristic
    high_risk = []
    for h in holdings_analysis:
        beta = h.get("beta")
        if beta is not None and beta >= 1.5:
            high_risk.append(h.get("ticker"))
    holdings_with_data = sum(1 for h in holdings_analysis if h.get("data_available"))
    holdings_without_data = len(holdings_analysis) - holdings_with_data

    # Sector breakdown
    sector_breakdown: dict[str, float] = {}
    for h in holdings_analysis:
        sector = h.get("sector", "Unknown")
        sector_breakdown[sector] = sector_breakdown.get(sector, 0.0) + float(h.get("allocation_pct", 0.0))

    # Overall risk level heuristic
    risk_level = "UNKNOWN"
    if weighted_avg_beta is not None:
        if weighted_avg_beta >= 1.3:
            risk_level = "HIGH"
        elif weighted_avg_beta >= 0.9:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

    return {
        "overall_risk_level": risk_level,
        "weighted_avg_beta": None if weighted_avg_beta is None else round(weighted_avg_beta, 3),
        "weighted_avg_pe": None if weighted_avg_pe is None else round(weighted_avg_pe, 2),
        "concentration_hhi": None if concentration_hhi is None else round(concentration_hhi, 4),
        "max_single_holding_pct": None if max_single is None else round(max_single, 2),
        "high_risk_holdings": high_risk,
        "holdings_with_data": holdings_with_data,
        "holdings_without_data": holdings_without_data,
        "sector_breakdown": {k: round(v, 2) for k, v in sector_breakdown.items()},
    }


# ----------------- Structured LLM signals -----------------
class HoldingSignal(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    signal: str = Field(description="BUY, HOLD, or SELL")


class HoldingSignals(BaseModel):
    signals: list[HoldingSignal]


def get_llm_signals(holdings_summary: list[dict], llm) -> list[dict] | None:
    """Ask the LLM for per-holding BUY/HOLD/SELL signals and combine with rules."""
    tickers_with_data = [
        h for h in holdings_summary
        if h.get("data_available", True) and h.get("rule_signal")
    ]
    if not tickers_with_data:
        return None

    prompt = (
        "For each stock below, give a BUY, HOLD, or SELL recommendation "
        "based on the technical and fundamental data provided.\n\n"
        + json.dumps(tickers_with_data, indent=2, default=str)
    )

    try:
        structured_llm = llm.with_structured_output(HoldingSignals)
        result = structured_llm.invoke([HumanMessage(content=prompt)])
        signal_map = {s.ticker.upper(): s.signal.upper() for s in result.signals}

        for h in holdings_summary:
            ticker = h["ticker"].upper()
            if ticker in signal_map and h.get("rule_signal"):
                h["llm_signal"] = signal_map[ticker]
                combined_rec, combined_score = combine_signals(h["rule_signal"], signal_map[ticker])
                h["combined_signal"] = combined_rec
                h["combined_score"] = round(combined_score, 3)

        return holdings_summary
    except Exception:
        return None


# ----------------- Graph nodes -----------------
def load_portfolio_node(state: AnalyticsState) -> dict:
    """Node 1: Load portfolio from state or disk."""
    if state.portfolio and state.portfolio.get("holdings"):
        return {"portfolio": state.portfolio}

    portfolio = _load_portfolio_file() or {}
    return {"portfolio": portfolio}


def build_benchmark_context(state: AnalyticsState) -> dict:
    """Node 2: Compare portfolio sectors against S&P 500."""
    holdings = state.portfolio.get("holdings", [])
    if not holdings:
        return {"benchmark_context": "", "sector_rows": []}

    sector_weights: dict[str, float] = {}
    lines = ["PORTFOLIO WEIGHT ANALYSIS DATA\n", "Portfolio Holdings:"]
    for h in holdings:
        fundamentals = fetch_fundamentals(h["ticker"])
        sector = fundamentals.get("sector", "Unknown")
        company = fundamentals.get("company_name", h["ticker"])
        sector_weights[sector] = sector_weights.get(sector, 0.0) + float(h["allocation_pct"])
        lines.append(f"  {h['ticker']} ({company}): {float(h['allocation_pct']):.1f}% - Sector: {sector}")

    lines.append("\nSector Allocation Comparison (Portfolio vs S&P 500 Benchmark):")
    all_sectors = sorted(set(list(SP500_SECTOR_WEIGHTS.keys()) + list(sector_weights.keys())))

    rows = []
    for sector in all_sectors:
        port_w = float(sector_weights.get(sector, 0.0))
        sp_w = float(SP500_SECTOR_WEIGHTS.get(sector, 0.0))
        diff = port_w - sp_w
        status = "OVERWEIGHT" if diff > 5 else "UNDERWEIGHT" if diff < -5 else "NEUTRAL"
        rows.append({"sector": sector, "portfolio_weight": port_w, "sp500_weight": sp_w, "diff": diff, "status": status})
        lines.append(
            f"  {sector}: Portfolio {port_w:.1f}% vs S&P 500 {sp_w:.1f}% (Difference: {diff:+.1f}%)"
            f" → {status}"
        )

    return {"benchmark_context": "\n".join(lines), "sector_rows": rows}


def analyze_holdings_node(state: AnalyticsState) -> dict:
    """Node 3: Analyze each holding."""
    holdings = state.portfolio.get("holdings", [])
    if not holdings:
        return {"holdings_analysis": []}

    analyses = []
    for h in holdings:
        try:
            analyses.append(analyze_holding(h["ticker"], float(h["allocation_pct"])))
        except Exception:
            analyses.append({"ticker": h.get("ticker", "?"), "allocation_pct": float(h.get("allocation_pct", 0.0)), "data_available": False})
    return {"holdings_analysis": analyses}


def compute_risk_node(state: AnalyticsState) -> dict:
    """Node 4: Compute portfolio-level risk metrics."""
    if not state.holdings_analysis:
        return {"risk_metrics": {"overall_risk_level": "UNKNOWN"}}
    return {"risk_metrics": compute_portfolio_risk(state.holdings_analysis)}


def generate_analysis(state: AnalyticsState) -> dict:
    """Node 5: LLM narrative analysis + optional structured per-holding signals."""
    if not state.portfolio:
        return {"messages": [SystemMessage(content="No portfolio data available.")]}

    holdings_summary = []
    for h in state.holdings_analysis:
        entry = {
            "ticker": h.get("ticker"),
            "allocation_pct": h.get("allocation_pct"),
            "sector": h.get("sector", "Unknown"),
        }
        if h.get("data_available"):
            entry.update({
                "current_price": h.get("current_price"),
                "rsi": round(h["rsi"], 1) if h.get("rsi") is not None else None,
                "macd": round(h["macd"], 4) if h.get("macd") is not None else None,
                "signal_line": round(h["signal_line"], 4) if h.get("signal_line") is not None else None,
                "sma_50": round(h["sma_50"], 2) if h.get("sma_50") is not None else None,
                "sma_200": round(h["sma_200"], 2) if h.get("sma_200") is not None else None,
                "pe_ratio": h.get("pe_ratio"),
                "pb_ratio": h.get("pb_ratio"),
                "roe": h.get("roe"),
                "beta": h.get("beta"),
                "dividend_yield": h.get("dividend_yield"),
                "total_score": h.get("total_score"),
                "rule_signal": h.get("rule_signal"),
            })
        else:
            entry["data_available"] = False
        holdings_summary.append(entry)

    context_parts = []
    if state.benchmark_context:
        context_parts.append(state.benchmark_context)
    context_parts.append("\nPER-HOLDING ANALYSIS:")
    context_parts.append(json.dumps(holdings_summary, indent=2, default=str))
    context_parts.append("\nPORTFOLIO RISK METRICS:")
    context_parts.append(json.dumps(state.risk_metrics, indent=2, default=str))
    full_context = "\n".join(context_parts)

    llm = load_llm_from_env()
    messages = [SystemMessage(content=ANALYSIS_PROMPT), HumanMessage(content=full_context)]
    response = llm.invoke(messages)

    updated_holdings = get_llm_signals(holdings_summary, llm)
    if updated_holdings:
        return {"messages": [response], "holdings_analysis": updated_holdings}
    return {"messages": [response]}


def build_graph():
    graph_builder = StateGraph(AnalyticsState)
    graph_builder.add_node("load_portfolio", load_portfolio_node)
    graph_builder.add_node("build_benchmark_context", build_benchmark_context)
    graph_builder.add_node("analyze_holdings", analyze_holdings_node)
    graph_builder.add_node("compute_risk", compute_risk_node)
    graph_builder.add_node("generate_analysis", generate_analysis)

    graph_builder.set_entry_point("load_portfolio")
    graph_builder.add_edge("load_portfolio", "build_benchmark_context")
    graph_builder.add_edge("build_benchmark_context", "analyze_holdings")
    graph_builder.add_edge("analyze_holdings", "compute_risk")
    graph_builder.add_edge("compute_risk", "generate_analysis")
    graph_builder.add_edge("generate_analysis", END)
    return graph_builder.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_analytics(portfolio: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the analytics pipeline and return the full result dict."""
    graph = get_graph()
    state = AnalyticsState(portfolio=portfolio or {})
    result = graph.invoke(state)
    # result is a dict-like state; convert for JSON compatibility
    out = {
        "portfolio": result.get("portfolio", {}),
        "benchmark_context": result.get("benchmark_context", ""),
        "sector_rows": result.get("sector_rows", []),
        "holdings_analysis": result.get("holdings_analysis", []),
        "risk_metrics": result.get("risk_metrics", {}),
        "analysis_report": (result.get("messages")[-1].content if result.get("messages") else ""),
    }
    return out
