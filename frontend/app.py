import os
import uuid
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialization ──────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile_loaded" not in st.session_state:
    st.session_state.profile_loaded = False
if "profile" not in st.session_state:
    st.session_state.profile = {}


# ── Helpers ────────────────────────────────────────────────────────────────────
def api(method: str, path: str, **kwargs):
    try:
        resp = getattr(requests, method)(f"{BACKEND_URL}{path}", timeout=180, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend. Make sure the FastAPI server is running.")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def ensure_session():
    if not st.session_state.session_id:
        data = api("post", "/sessions", json={"user_id": st.session_state.user_id})
        if data:
            st.session_state.session_id = data["session_id"]


def load_profile():
    if not st.session_state.profile_loaded:
        data = api("get", f"/profile/{st.session_state.user_id}")
        if data:
            st.session_state.profile = data
            st.session_state.profile_loaded = True


def agent_badge_color(agent: str) -> str:
    colors = {
        "Portfolio Analyst": "#1f77b4",
        "Market Trends Analyst": "#2ca02c",
        "Tax Educator": "#d62728",
        "Finance Q&A": "#9467bd",
        "Goal Planning Advisor": "#8c564b",
        "News Synthesizer": "#e377c2",
        "Router": "#7f7f7f",
    }
    return colors.get(agent, "#17becf")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💼 Financial Advisor")
    st.caption(f"User ID: `{st.session_state.user_id[:8]}…`")

    st.divider()

    # ── Profile ────────────────────────────────────────────────────────────────
    load_profile()
    profile = st.session_state.profile

    with st.expander("👤 Your Profile", expanded=not st.session_state.profile_loaded):
        with st.form("profile_form"):
            name = st.text_input("Name", value=profile.get("name") or "")
            risk = st.selectbox(
                "Risk Tolerance",
                ["low", "medium", "high"],
                index=["low", "medium", "high"].index(profile.get("risk_tolerance") or "medium"),
            )
            horizon = st.selectbox(
                "Investment Horizon",
                ["short", "medium", "long"],
                index=["short", "medium", "long"].index(profile.get("investment_horizon") or "medium"),
            )
            goal = st.selectbox(
                "Primary Goal",
                ["growth", "income", "preservation"],
                index=["growth", "income", "preservation"].index(profile.get("primary_goal") or "growth"),
            )
            bracket = st.selectbox(
                "Net Worth Bracket",
                ["<50k", "50k-250k", "250k-1m", ">1m"],
                index=["<50k", "50k-250k", "250k-1m", ">1m"].index(profile.get("net_worth_bracket") or "<50k"),
            )
            tax_status = st.selectbox(
                "Tax Filing Status",
                ["single", "married_filing_jointly", "married_filing_separately", "head_of_household"],
                index=["single", "married_filing_jointly", "married_filing_separately", "head_of_household"].index(
                    profile.get("tax_filing_status") or "single"
                ),
            )
            tickers_raw = st.text_input(
                "Portfolio Tickers (comma-separated)",
                value=", ".join(profile.get("tickers") or []),
                placeholder="e.g. AAPL, MSFT, SPY",
            )

            if st.form_submit_button("Save Profile", use_container_width=True):
                tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
                payload = {
                    "name": name,
                    "risk_tolerance": risk,
                    "investment_horizon": horizon,
                    "primary_goal": goal,
                    "net_worth_bracket": bracket,
                    "tax_filing_status": tax_status,
                    "tickers": tickers,
                }
                result = api("put", f"/profile/{st.session_state.user_id}", json=payload)
                if result:
                    st.session_state.profile = payload
                    st.success("Profile saved!")

    st.divider()

    # ── Conversation controls ──────────────────────────────────────────────────
    if st.button("➕ New Conversation", use_container_width=True):
        data = api("post", "/sessions", json={"user_id": st.session_state.user_id})
        if data:
            st.session_state.session_id = data["session_id"]
            st.session_state.messages = []
            st.rerun()

    st.subheader("Recent Conversations")
    sessions = api("get", f"/sessions/{st.session_state.user_id}") or []
    for s in sessions[:10]:
        label = s.get("title", "Conversation")[:40]
        is_active = s["session_id"] == st.session_state.session_id
        btn_label = f"{'▶ ' if is_active else ''}{label}"
        if st.button(btn_label, key=s["session_id"], use_container_width=True):
            st.session_state.session_id = s["session_id"]
            history = api("get", f"/history/{s['session_id']}") or []
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"], "agent_used": m.get("agent_used")}
                for m in history
            ]
            st.rerun()

    st.divider()
    st.caption("⚠️ This tool provides financial **education**, not personalized financial advice. Consult a licensed advisor before making investment decisions.")


# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("Your AI Financial Advisor")
st.caption("Ask me about your portfolio, market trends, taxes, financial goals, or any finance question.")

ensure_session()

# Display existing messages
for msg in st.session_state.messages:
    role = msg["role"]
    display_role = "user" if role == "human" else "assistant"
    with st.chat_message(display_role):
        st.markdown(msg["content"])
        if role == "assistant" and msg.get("agent_used"):
            color = agent_badge_color(msg["agent_used"])
            st.markdown(
                f'<span style="background-color:{color};color:white;padding:2px 8px;border-radius:10px;font-size:0.75em">{msg["agent_used"]}</span>',
                unsafe_allow_html=True,
            )

# Chat input
if prompt := st.chat_input("Ask me anything about your finances…"):
    st.session_state.messages.append({"role": "human", "content": prompt, "agent_used": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            result = api(
                "post",
                "/chat",
                json={
                    "user_id": st.session_state.user_id,
                    "session_id": st.session_state.session_id,
                    "message": prompt,
                },
            )

        if result:
            response_text = result["response"]
            agent_used = result.get("agent_used", "")
            st.markdown(response_text)
            if agent_used:
                color = agent_badge_color(agent_used)
                st.markdown(
                    f'<span style="background-color:{color};color:white;padding:2px 8px;border-radius:10px;font-size:0.75em">{agent_used}</span>',
                    unsafe_allow_html=True,
                )
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text, "agent_used": agent_used}
            )
        else:
            st.error("Failed to get a response. Please check the backend and try again.")
