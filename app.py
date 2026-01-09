import streamlit as st
from therabot_core import get_response, SYSTEM_PROMPT

st.set_page_config(
    page_title="TheraBot",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 TheraBot")
st.caption("A safe, empathetic space to talk")

if "conversation" not in st.session_state:
    st.session_state.conversation = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

for msg in st.session_state.conversation[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.conversation.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("TheraBot is thinking..."):
        reply = get_response(
            st.session_state.conversation,
            user_input
        )

    st.session_state.conversation.append({
        "role": "assistant",
        "content": reply
    })

    st.rerun()
