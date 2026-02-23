"""
Shared utilities for the Streamlit Investment Coach app.
- Loads environment variables
- Builds LLM instances
- Provides portfolio persistence helpers
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# ---- Paths ----
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PORTFOLIO_PATH = DATA_DIR / "portfolio.json"
ANALYTICS_CACHE_PATH = DATA_DIR / "analytics_result.json"


def load_env() -> None:
    """Load .env into process env (no-op if already loaded)."""
    load_dotenv(override=False)


def load_llm_from_env():
    """
    Create a LangChain chat model using environment variables.

    Env vars:
      - OPENAI_API_KEY (required)
      - OPENAI_MODEL (optional; default: gpt-4o-mini)
      - OPENAI_TEMPERATURE (optional; default: 0.2)
    """
    load_env()
    from langchain_openai import ChatOpenAI  # lazy import

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    except ValueError:
        temperature = 0.2

    return ChatOpenAI(model=model, temperature=temperature)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_portfolio(portfolio: Dict[str, Any]) -> None:
    save_json(PORTFOLIO_PATH, portfolio)


def load_portfolio() -> Optional[Dict[str, Any]]:
    return load_json(PORTFOLIO_PATH)


def save_analytics_result(result: Dict[str, Any]) -> None:
    save_json(ANALYTICS_CACHE_PATH, result)


def load_analytics_result() -> Optional[Dict[str, Any]]:
    return load_json(ANALYTICS_CACHE_PATH)
