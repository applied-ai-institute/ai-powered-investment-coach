import streamlit as st
import pandas as pd

from utils import load_env, load_portfolio, save_portfolio, load_analytics_result, save_analytics_result
from agents.portfolio_builder import chat as portfolio_chat
from agents.analytics import run_analytics
from agents.education import answer as education_answer

from langchain_core.messages import HumanMessage, AIMessage


APP_TITLE = "Sequential Multi‑Agent Investment Coach"


def _init_session_state():
    if "page" not in st.session_state:
        st.session_state.page = "Portfolio Builder"
    if "pb_thread_id" not in st.session_state:
        st.session_state.pb_thread_id = "streamlit"
    if "pb_chat" not in st.session_state:
        st.session_state.pb_chat = []  # list[{"role": "...", "content": "..."}]
    if "edu_chat" not in st.session_state:
        st.session_state.edu_chat = []
    if "last_portfolio" not in st.session_state:
        st.session_state.last_portfolio = load_portfolio() or {}
    if "analytics_result" not in st.session_state:
        st.session_state.analytics_result = load_analytics_result() or {}


def _sidebar_nav():
    with st.sidebar:
        st.markdown(f"## {APP_TITLE}")
        st.caption("Educational demo — not financial advice.")

        pages = ["Portfolio Builder", "Analytics Dashboard", "Investment Education", "Settings"]
        st.session_state.page = st.radio("Navigate", pages, index=pages.index(st.session_state.page))

        st.divider()
        portfolio = load_portfolio()
        if portfolio and portfolio.get("holdings"):
            st.success(f"Portfolio loaded: **{portfolio.get('name','(unnamed)')}**")
            st.caption(f"{len(portfolio['holdings'])} holdings")
        else:
            st.warning("No portfolio saved yet. Build one first.")

        if st.button("Clear local portfolio + analytics cache"):
            save_portfolio({})
            save_analytics_result({})
            st.session_state.last_portfolio = {}
            st.session_state.analytics_result = {}
            st.toast("Cleared.", icon="🧹")


def page_portfolio_builder():
    st.header("Portfolio Builder")
    st.write("Chat with the **Portfolio Builder agent** to create a portfolio. It can use web search + a portfolio generation tool.")

    # show existing
    for m in st.session_state.pb_chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_text = st.chat_input("Describe your goals, horizon, and risk tolerance…")
    if user_text:
        st.session_state.pb_chat.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = portfolio_chat(user_text, thread_id=st.session_state.pb_thread_id)
                st.markdown(reply)
        st.session_state.pb_chat.append({"role": "assistant", "content": reply})

        # refresh portfolio view after any tool call
        st.session_state.last_portfolio = load_portfolio() or {}

    st.divider()
    st.subheader("Current saved portfolio")
    portfolio = st.session_state.last_portfolio
    if portfolio and portfolio.get("holdings"):
        st.markdown(f"**{portfolio.get('name','(unnamed)')}** — {portfolio.get('description','')}")
        df = pd.DataFrame(portfolio["holdings"])
        # nice column order if present
        cols = [c for c in ["ticker","company_name","investment_type","allocation_pct","rationale"] if c in df.columns] + [c for c in df.columns if c not in {"ticker","company_name","investment_type","allocation_pct","rationale"}]
        st.dataframe(df[cols], use_container_width=True)
        total = df["allocation_pct"].astype(float).sum() if "allocation_pct" in df.columns else None
        if total is not None:
            st.caption(f"Total allocation: {total:.1f}%")
    else:
        st.info("No portfolio saved yet. Ask the agent to generate one (it will call the `portfolio_generation` tool).")


def page_analytics_dashboard():
    st.header("Analytics Dashboard")
    st.write("Run the **Portfolio Analytics agent** to fetch live market data, compute rule-based scores + risk metrics, and generate an analysis report.")

    portfolio = load_portfolio() or {}
    if not (portfolio and portfolio.get("holdings")):
        st.warning("Build and save a portfolio first on the Portfolio Builder page.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Run analytics", type="primary"):
            with st.spinner("Fetching market data and running analysis…"):
                result = run_analytics(portfolio=portfolio)
                st.session_state.analytics_result = result
                save_analytics_result(result)
                st.success("Analytics updated.")
    with col2:
        st.caption("Tip: Market data comes from yfinance and may have occasional gaps.")

    result = st.session_state.analytics_result or {}
    if not result:
        st.info("Click **Run analytics** to generate results.")
        return

    st.subheader("Portfolio risk metrics")
    rm = result.get("risk_metrics", {}) or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk level", rm.get("overall_risk_level", "N/A"))
    c2.metric("Weighted beta", rm.get("weighted_avg_beta", "N/A"))
    c3.metric("Concentration (HHI)", rm.get("concentration_hhi", "N/A"))
    c4.metric("Max holding", f"{rm.get('max_single_holding_pct','N/A')}%")

    st.subheader("Sector comparison vs S&P 500")
    sector_rows = result.get("sector_rows") or []
    if sector_rows:
        sdf = pd.DataFrame(sector_rows)
        st.dataframe(sdf, use_container_width=True)
    else:
        st.caption("No sector rows available.")

    st.subheader("Per-holding analysis")
    ha = result.get("holdings_analysis") or []
    if ha:
        hdf = pd.DataFrame(ha)
        # keep it readable
        keep = [c for c in [
            "ticker","allocation_pct","sector","current_price","rsi","pe_ratio","beta",
            "total_score","rule_signal","llm_signal","combined_signal","combined_score","data_available"
        ] if c in hdf.columns]
        st.dataframe(hdf[keep], use_container_width=True)

        # simple charts
        try:
            alloc_df = pd.DataFrame(portfolio["holdings"])[["ticker","allocation_pct"]].copy()
            alloc_df["allocation_pct"] = alloc_df["allocation_pct"].astype(float)
            st.bar_chart(alloc_df.set_index("ticker")["allocation_pct"])
        except Exception:
            pass
    else:
        st.caption("No holdings analysis available.")

    st.subheader("LLM analysis report")
    report = result.get("analysis_report", "")
    if report:
        st.markdown(report)
    else:
        st.caption("No report generated.")


def page_investment_education():
    st.header("Investment Education")
    st.write("Ask questions to the **Investment Education agent**. It personalizes answers using your saved portfolio and (if available) the latest analytics.")

    portfolio = load_portfolio() or {}
    analytics = st.session_state.analytics_result or load_analytics_result() or {}

    for m in st.session_state.edu_chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_text = st.chat_input("Ask an investing question (e.g., 'What is RSI?' or 'How risky is my portfolio?')…")
    if user_text:
        st.session_state.edu_chat.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # convert history to BaseMessages for the agent
        history_msgs = []
        for m in st.session_state.edu_chat[:-1]:
            if m["role"] == "user":
                history_msgs.append(HumanMessage(content=m["content"]))
            else:
                history_msgs.append(AIMessage(content=m["content"]))

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = education_answer(
                    question=user_text,
                    history=history_msgs,
                    portfolio=portfolio,
                    analytics=analytics,
                )
                st.markdown(reply)

        st.session_state.edu_chat.append({"role": "assistant", "content": reply})


def page_settings():
    st.header("Settings")
    st.write("Environment variables are loaded from a local `.env` file (not committed).")
    st.code(
        "OPENAI_API_KEY=...\n"
        "OPENAI_MODEL=gpt-4o-mini\n"
        "OPENAI_TEMPERATURE=0.2\n"
        "SERPER_API_KEY=...\n",
        language="bash",
    )

    st.subheader("Diagnostics")
    load_env()
    import os
    st.write({
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "OPENAI_TEMPERATURE": os.getenv("OPENAI_TEMPERATURE", "0.2"),
        "SERPER_API_KEY_set": bool(os.getenv("SERPER_API_KEY")),
        "portfolio_saved": bool((load_portfolio() or {}).get("holdings")),
        "analytics_cached": bool(st.session_state.analytics_result),
    })


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _init_session_state()
    _sidebar_nav()

    page = st.session_state.page
    if page == "Portfolio Builder":
        page_portfolio_builder()
    elif page == "Analytics Dashboard":
        page_analytics_dashboard()
    elif page == "Investment Education":
        page_investment_education()
    else:
        page_settings()


if __name__ == "__main__":
    main()
