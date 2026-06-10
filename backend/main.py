import logging
import os
import traceback
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.orm import Session

load_dotenv()

from backend.db.database import get_db, init_db
from backend.db.crud import (
    create_session,
    get_or_create_user,
    get_session_messages,
    get_user_sessions,
    save_message,
    update_session_title,
    update_user_profile,
)
from backend.models.schemas import ChatRequest, ChatResponse, SessionCreate, UserProfile
from backend.agents.graph import graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Pre-warm the vector store so the first request doesn't block on
    # the HuggingFace model download + ChromaDB seeding (can take 60–90s).
    import asyncio
    from backend.rag.retriever import get_vectorstore
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_vectorstore)
    yield


app = FastAPI(title="Financial Advisor API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    user = get_or_create_user(db, request.user_id)

    profile = {
        "user_id": user.user_id,
        "name": user.name,
        "risk_tolerance": user.risk_tolerance,
        "investment_horizon": user.investment_horizon,
        "primary_goal": user.primary_goal,
        "net_worth_bracket": user.net_worth_bracket,
        "tax_filing_status": user.tax_filing_status,
        "tickers": user.tickers or [],
    }

    prev_messages = get_session_messages(db, request.session_id)
    lc_messages = []
    for msg in prev_messages[-10:]:
        if msg.role == "human":
            lc_messages.append(HumanMessage(content=msg.content))
        else:
            lc_messages.append(AIMessage(content=msg.content))
    lc_messages.append(HumanMessage(content=request.message))

    initial_state = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "user_profile": profile,
        "messages": lc_messages,
        "current_query": request.message,
        "classified_intent": None,
        "retrieved_context": [],
        "yfinance_data": None,
        "agent_outputs": {},
        "final_response": "",
        "agent_used": "",
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        logger.error("Graph invocation failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {exc}")

    final_response = result.get("final_response") or "I wasn't able to process your request. Please try again."
    agent_used = result.get("agent_used") or "Unknown"

    # Auto-title the session on the first message
    if len(prev_messages) == 0:
        title = request.message[:60] + ("…" if len(request.message) > 60 else "")
        update_session_title(db, request.session_id, title)

    save_message(db, request.session_id, request.user_id, "human", request.message)
    save_message(db, request.session_id, request.user_id, "assistant", final_response, agent_used)

    return ChatResponse(response=final_response, agent_used=agent_used, session_id=request.session_id)


@app.get("/profile/{user_id}")
async def get_profile(user_id: str, db: Session = Depends(get_db)):
    user = get_or_create_user(db, user_id)
    return {
        "user_id": user.user_id,
        "name": user.name,
        "risk_tolerance": user.risk_tolerance,
        "investment_horizon": user.investment_horizon,
        "primary_goal": user.primary_goal,
        "net_worth_bracket": user.net_worth_bracket,
        "tax_filing_status": user.tax_filing_status,
        "tickers": user.tickers or [],
    }


@app.put("/profile/{user_id}")
async def update_profile(user_id: str, profile: UserProfile, db: Session = Depends(get_db)):
    data = profile.model_dump(exclude_none=True)
    user = update_user_profile(db, user_id, data)
    return {"message": "Profile updated", "user_id": user.user_id}


@app.post("/sessions")
async def create_new_session(data: SessionCreate, db: Session = Depends(get_db)):
    get_or_create_user(db, data.user_id)
    session = create_session(db, data.user_id, data.title or "New Conversation")
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
    }


@app.get("/sessions/{user_id}")
async def list_sessions(user_id: str, db: Session = Depends(get_db)):
    sessions = get_user_sessions(db, user_id)
    return [
        {
            "session_id": s.session_id,
            "title": s.title,
            "created_at": s.created_at.isoformat(),
        }
        for s in sessions
    ]


@app.get("/history/{session_id}")
async def get_history(session_id: str, db: Session = Depends(get_db)):
    messages = get_session_messages(db, session_id)
    return [
        {
            "message_id": m.message_id,
            "role": m.role,
            "content": m.content,
            "agent_used": m.agent_used,
            "timestamp": m.timestamp.isoformat(),
        }
        for m in messages
    ]


@app.get("/health")
async def health():
    return {"status": "ok"}
