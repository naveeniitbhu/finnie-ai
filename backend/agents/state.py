from typing import Any, Dict, List, Optional
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages


class ConversationState(TypedDict, total=False):
    user_id: str
    session_id: str
    user_profile: Dict[str, Any]
    messages: Annotated[list, add_messages]
    current_query: str
    classified_intent: Optional[Dict[str, Any]]
    retrieved_context: List[str]
    yfinance_data: Optional[Dict[str, Any]]
    agent_outputs: Dict[str, str]
    final_response: str
    agent_used: str
