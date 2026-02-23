"""
Investment Education agent implemented with LangGraph.

Purpose: Answer investment questions, personalized to the user's current portfolio.
"""
from __future__ import annotations

import yfinance as yf
from typing import Annotated, Sequence, Optional, Dict, Any, List

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from utils import load_llm_from_env, load_portfolio as _load_portfolio_file

# Same benchmark weights as Analytics agent
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

EDUCATION_PROMPT = """You are an expert investment education assistant. Your role is to help users understand investment concepts, portfolio management, and financial markets.

Guidelines:
- Explain concepts clearly using plain language.
- Use real-world examples when possible.
- If the user has a portfolio, reference their specific holdings to make explanations more relevant and personalized.
- Cover topics like: asset allocation, diversification, risk management, technical analysis (RSI, MACD, SMA), fundamental analysis (P/E, P/B, ROE), sector rotation, rebalancing, dividend investing, growth vs value.
- Always remind users that this is for educational purposes only and not financial advice.
- When discussing the user's portfolio, explain the "why" behind observations (e.g., why being overweight in tech can increase volatility).
- Be conversational and encourage follow-up questions.
"""


class EducationState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = []
    system_prompt: str = EDUCATION_PROMPT
    portfolio_context: str = ""
    portfolio: dict = Field(default_factory=dict)
    analytics: dict = Field(default_factory=dict)


def _fetch_sector(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector") or "Unknown"
    except Exception:
        return "Unknown"


def compute_sector_comparison(holdings: list[dict]) -> str:
    """Compare portfolio sector weights against S&P 500."""
    if not holdings:
        return ""

    sector_weights: dict[str, float] = {}
    for h in holdings:
        sector = _fetch_sector(h["ticker"])
        sector_weights[sector] = sector_weights.get(sector, 0.0) + float(h.get("allocation_pct", 0.0))

    lines = ["SECTOR COMPARISON (Portfolio vs S&P 500 benchmark):"]
    all_sectors = sorted(set(list(SP500_SECTOR_WEIGHTS.keys()) + list(sector_weights.keys())))
    for sector in all_sectors:
        port_w = float(sector_weights.get(sector, 0.0))
        sp_w = float(SP500_SECTOR_WEIGHTS.get(sector, 0.0))
        diff = port_w - sp_w
        status = "OVERWEIGHT" if diff > 5 else "UNDERWEIGHT" if diff < -5 else "NEUTRAL"
        lines.append(f"- {sector}: Portfolio {port_w:.1f}% vs S&P 500 {sp_w:.1f}% ({diff:+.1f}%) → {status}")

    lines.append("\nOverweight = more than S&P 500 benchmark; underweight = less.")
    return "\n".join(lines)


def enrich_context(state: EducationState) -> dict:
    """Node 1: Build enriched system prompt with portfolio + analytics context."""
    portfolio = state.portfolio if state.portfolio.get("holdings") else (_load_portfolio_file() or {})
    portfolio_context = ""

    if portfolio and portfolio.get("holdings"):
        holdings = portfolio["holdings"]
        tickers = [h["ticker"] for h in holdings]
        portfolio_context = f"USER'S CURRENT PORTFOLIO:\nSecurities: {', '.join(tickers)}.\n"

        allocations = [
            f"{h['ticker']} ({h.get('allocation_pct', 0)}%)"
            for h in holdings if h.get("allocation_pct") is not None
        ]
        if allocations:
            portfolio_context += f"Allocations: {', '.join(allocations)}.\n"

        sector_text = compute_sector_comparison(
            [{"ticker": h["ticker"], "allocation_pct": h.get("allocation_pct", 0)} for h in holdings]
        )
        if sector_text:
            portfolio_context += f"\n{sector_text}\n"

    # Add analytics highlights if present
    analytics_context = ""
    if state.analytics:
        rm = state.analytics.get("risk_metrics") or {}
        if rm:
            analytics_context += (
                "\nRECENT ANALYTICS HIGHLIGHTS:\n"
                f"- Overall risk level: {rm.get('overall_risk_level', 'N/A')}\n"
                f"- Weighted avg beta: {rm.get('weighted_avg_beta', 'N/A')}\n"
                f"- Concentration (HHI): {rm.get('concentration_hhi', 'N/A')}\n"
                f"- Max single holding: {rm.get('max_single_holding_pct', 'N/A')}%\n"
            )

        ha = state.analytics.get("holdings_analysis") or []
        if ha:
            # Include top 5 by allocation with combined signals if available
            top = sorted(ha, key=lambda x: float(x.get("allocation_pct", 0.0)), reverse=True)[:5]
            parts = []
            for h in top:
                sig = h.get("combined_signal") or h.get("rule_signal") or "-"
                parts.append(f"{h.get('ticker')} ({h.get('allocation_pct', 0)}%): {sig}")
            analytics_context += "\nTop holdings signals: " + ", ".join(parts) + "\n"

    full_prompt = EDUCATION_PROMPT
    if portfolio_context:
        full_prompt += "\n\n" + portfolio_context
    if analytics_context:
        full_prompt += "\n" + analytics_context

    return {"system_prompt": full_prompt, "portfolio_context": portfolio_context, "portfolio": portfolio}


def call_model(state: EducationState) -> dict:
    """Node 2: Call LLM with system prompt + conversation history."""
    llm = load_llm_from_env()
    messages_for_llm = [SystemMessage(content=state.system_prompt)] + list(state.messages)
    response = llm.invoke(messages_for_llm)
    return {"messages": [response]}


def build_graph():
    graph_builder = StateGraph(EducationState)
    graph_builder.add_node("enrich_context", enrich_context)
    graph_builder.add_node("call_model", call_model)

    graph_builder.set_entry_point("enrich_context")
    graph_builder.add_edge("enrich_context", "call_model")
    graph_builder.add_edge("call_model", END)
    return graph_builder.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def answer(question: str, history: List[BaseMessage], portfolio: Optional[Dict[str, Any]] = None, analytics: Optional[Dict[str, Any]] = None) -> str:
    """Answer a question given chat history + optional portfolio/analytics context."""
    graph = get_graph()
    state = EducationState(messages=history + [HumanMessage(content=question)], portfolio=portfolio or {}, analytics=analytics or {})
    result = graph.invoke(state)
    return result["messages"][-1].content if result.get("messages") else ""
