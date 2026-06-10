# Finnie AI — Multi-Agent Financial Advisor

A conversational financial advisory application powered by a multi-agent AI pipeline. Ask questions about your portfolio, market trends, taxes, financial goals, or recent financial news.

## Architecture

- **Frontend** — Streamlit chat interface with user profile management and conversation history
- **Backend** — FastAPI REST API orchestrating a LangGraph agent pipeline
- **Agents** — 6 specialist agents (Portfolio Analyst, Market Trends, Tax Educator, Finance Q&A, Goal Planning, News Synthesizer) routed by an LLM-based intent classifier
- **RAG** — ChromaDB vector store with HuggingFace embeddings for grounded responses
- **Live data** — yfinance for real-time market prices and financial news
- **Persistence** — SQLite for user profiles and conversation history

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/naveeniitbhu/finnie-ai.git
cd finnie-ai
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your API key for the active provider:

```
LLM_PROVIDER=google          # anthropic | google | openai
LLM_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=your_key_here
```

## Running the application

Open two terminals with the virtual environment activated.

**Terminal 1 — Backend**
```bash
uvicorn backend.main:app --reload --port 8000
```

Wait for the startup message confirming the vector store is ready before proceeding.

**Terminal 2 — Frontend**
```bash
streamlit run frontend/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Switching LLM providers

No code changes needed — update two lines in `.env`:

| Provider | `LLM_PROVIDER` | `LLM_MODEL` | Key variable |
|---|---|---|---|
| Google | `google` | `gemini-2.5-flash` | `GOOGLE_API_KEY` |
| Anthropic | `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `gpt-4o` | `OPENAI_API_KEY` |
