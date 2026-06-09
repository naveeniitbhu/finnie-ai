from pydantic import BaseModel
from typing import Optional, List


class UserProfile(BaseModel):
    name: Optional[str] = None
    risk_tolerance: Optional[str] = None
    investment_horizon: Optional[str] = None
    primary_goal: Optional[str] = None
    net_worth_bracket: Optional[str] = None
    tax_filing_status: Optional[str] = None
    tickers: Optional[List[str]] = []


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    agent_used: str
    session_id: str


class SessionCreate(BaseModel):
    user_id: str
    title: Optional[str] = "New Conversation"


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    title: str
    created_at: str

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    agent_used: Optional[str] = None
    timestamp: str

    class Config:
        from_attributes = True
