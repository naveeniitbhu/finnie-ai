# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the application

Two processes must run simultaneously — backend first, then frontend:

```bash
# Terminal 1 — FastAPI backend (run from project root)
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Streamlit frontend
streamlit run frontend/app.py
```

Install dependencies before first run:
```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the API key for the active provider before starting.

## Environment configuration

LLM provider is controlled entirely via `.env` — no code changes needed to switch:

```
LLM_PROVIDER=google          # anthropic | google | openai
LLM_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=...
```

`backend/llm.py` is the single factory — all agents call `get_llm(temperature=X)` from there. Provider packages are lazily imported inside `get_llm()`, so only the active provider's package needs to be installed.

## Architecture

### Request flow

```
Streamlit (frontend/app.py)
  → POST /chat (FastAPI, backend/main.py)
    → LangGraph graph (backend/agents/graph.py)
        router_node → rag_node → [yfinance_node] → agent_executor_node → END
                                ↘ clarification_node → END
```

### LangGraph graph (backend/agents/graph.py)

The graph is compiled once at module load (`graph = build_graph()`). No checkpointer — state is rebuilt from SQLite on every request. The graph has four nodes:

- **router** — LLM call using `.with_structured_output(RouterIntent)` that returns `{primary_agent, chain, needs_yfinance, clarification_needed}`. Falls back to `finance_qa` silently on any failure.
- **rag** — retrieves up to 4 chunks from ChromaDB, domain-matched to the classified agent
- **yfinance** (conditional) — fetches live data only when `needs_yfinance=true`; each fetch is wrapped in a 12-second `ThreadPoolExecutor` timeout
- **agent_executor** — calls agent functions sequentially for chains; each function has signature `run_<agent>(state, prior_outputs) -> str`

### Agents (backend/agents/)

Agents are **plain Python functions**, not LangGraph nodes. `agent_executor_node` in `graph.py` dispatches to them. To add a new agent:
1. Create `backend/agents/<name>.py` with a `run_<name>(state, prior_outputs) -> str` function
2. Register it in `AGENT_FUNCTIONS` and `AGENT_LABELS` in `graph.py`
3. Add its description to `AGENT_DESCRIPTIONS` in `router.py`
4. Add seed documents with `"domain": "<name>"` metadata in `retriever.py`

### RAG (backend/rag/retriever.py)

ChromaDB with HuggingFace `all-MiniLM-L6-v2` embeddings. The vectorstore is a module-level singleton initialized lazily on first access. On first run it seeds 21 financial knowledge documents. FastAPI's `lifespan` pre-warms it at startup (via `run_in_executor`) to avoid blocking the first request.

The `data/chroma_db/` and `data/*.db` directories are git-ignored and created automatically at runtime.

### Persistence (backend/db/)

SQLite via SQLAlchemy. Three tables: `users`, `sessions`, `messages`. The DB file lives at `data/financial_advisor.db`. Sessions are auto-titled from the first 60 characters of the opening message.

### LLM provider switching

To switch providers, change two lines in `.env`. The factory in `backend/llm.py` handles the rest. The router uses `.with_structured_output(RouterIntent)` which relies on tool-calling — all supported providers (Anthropic, Google, OpenAI) implement this.
