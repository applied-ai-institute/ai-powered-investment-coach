"""
Portfolio Builder (Designer) agent implemented with LangGraph.

This module preserves the notebook's sequential tool-using chat agent:
- Tool 1: web search (Google Serper)
- Tool 2: portfolio_generation (validates + persists portfolio.json)
"""
from __future__ import annotations

import os
from typing import Literal, List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from utils import load_llm_from_env, save_portfolio

SYSTEM_PROMPT = """You are an Investment Coach assistant that helps users design an investment portfolio.

Follow this sequential process:
1) Understand the investor profile: goals, time horizon, risk tolerance, income stability, and any constraints.
2) Gather context: Use the search_web tool to look up current market information, ETF options, broad asset-class context, and diversification ideas.
3) Propose a portfolio: Use the portfolio_generation tool to generate a structured portfolio with 6–12 holdings across appropriate asset types (stocks, ETFs, bonds, cash-like).
4) Explain rationale: Provide concise reasoning and invite iteration based on feedback.

Rules:
- Ask clarifying questions if key profile info is missing.
- When searching, summarize findings concisely.
- Allocations should sum to ~100%.
- Always include a brief disclaimer that this is educational, not financial advice.
"""


@tool
def search_web(query: str) -> str:
    """Search the web for current market data, ETF information, sector performance, or investment news."""
    try:
        # Requires SERPER_API_KEY in env
        search = GoogleSerperAPIWrapper()
        return search.run(query)
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def portfolio_generation(
    portfolio_name: str,
    description: str,
    holdings: List[Dict[str, Any]],
) -> str:
    """Generate and save a structured investment portfolio.

    Args:
        portfolio_name: A descriptive name for the portfolio.
        description: A brief description of the portfolio strategy.
        holdings: List of dicts, each with: ticker, company_name, allocation_pct (0-100),
                  investment_type, and rationale.
    """
    if not holdings:
        return "Holdings list is empty. Please provide at least one holding."

    required_fields = ["ticker", "company_name", "allocation_pct", "investment_type", "rationale"]
    for i, h in enumerate(holdings):
        for field in required_fields:
            if field not in h:
                return f"Holding {i} is missing required field: '{field}'."

    # Validate allocations
    try:
        allocations = [float(h["allocation_pct"]) for h in holdings]
    except Exception:
        return "One or more allocation_pct values are not numeric."

    total_alloc = sum(allocations)
    if total_alloc < 95 or total_alloc > 105:
        return f"Allocations must sum to ~100%. Current total is {total_alloc:.1f}%."

    # Normalize small rounding drift to 100
    if abs(total_alloc - 100.0) > 0.01:
        for h in holdings:
            h["allocation_pct"] = float(h["allocation_pct"]) * 100.0 / total_alloc

    portfolio = {
        "name": portfolio_name,
        "description": description,
        "holdings": holdings,
    }
    save_portfolio(portfolio)

    tickers = [h["ticker"] for h in holdings]
    total_alloc = sum(float(h["allocation_pct"]) for h in holdings)
    return (
        f"Portfolio '{portfolio_name}' saved with {len(holdings)} holdings "
        f"({', '.join(tickers)}). Total allocation: {total_alloc:.1f}%."
    )


TOOLS = [search_web, portfolio_generation]


def build_graph():
    llm = load_llm_from_env()
    llm_with_tools = llm.bind_tools(TOOLS)

    def assistant(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
        return "tools" if state["messages"][-1].tool_calls else "__end__"

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("assistant", assistant)
    graph_builder.add_node("tools", ToolNode(TOOLS))
    graph_builder.add_edge(START, "assistant")
    graph_builder.add_conditional_edges("assistant", should_continue, ["tools", "__end__"])
    graph_builder.add_edge("tools", "assistant")

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def chat(user_input: str, thread_id: str = "default") -> str:
    """Run one chat turn and return the assistant's final message content."""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = None
    for event in graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="values",
    ):
        result = event

    return result["messages"][-1].content if result and result.get("messages") else ""
