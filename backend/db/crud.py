import uuid
from sqlalchemy.orm import Session
from .database import User, ConversationSession, Message


def get_or_create_user(db: Session, user_id: str) -> User:
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        user = User(user_id=user_id, tickers=[])
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def update_user_profile(db: Session, user_id: str, profile_data: dict) -> User:
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        user = User(user_id=user_id, **profile_data)
        db.add(user)
    else:
        for key, value in profile_data.items():
            setattr(user, key, value)
    db.commit()
    db.refresh(user)
    return user


def create_session(db: Session, user_id: str, title: str = "New Conversation") -> ConversationSession:
    session = ConversationSession(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        title=title,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_user_sessions(db: Session, user_id: str) -> list:
    return (
        db.query(ConversationSession)
        .filter(ConversationSession.user_id == user_id)
        .order_by(ConversationSession.created_at.desc())
        .all()
    )


def save_message(
    db: Session,
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    agent_used: str = None,
) -> Message:
    msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        user_id=user_id,
        role=role,
        content=content,
        agent_used=agent_used,
    )
    db.add(msg)
    db.commit()
    return msg


def get_session_messages(db: Session, session_id: str) -> list:
    return (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.timestamp.asc())
        .all()
    )


def update_session_title(db: Session, session_id: str, title: str):
    session = db.query(ConversationSession).filter(ConversationSession.session_id == session_id).first()
    if session:
        session.title = title
        db.commit()
