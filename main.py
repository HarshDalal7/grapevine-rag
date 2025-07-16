import streamlit as st
from rag_system import RAGSystem

st.set_page_config(page_title="Grapevine Q&A", page_icon="🌱", layout="wide")

# ----- Custom CSS -----
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: Arial, Helvetica, sans-serif;
        background: linear-gradient(135deg, #ffe259 0%, #ffa751 100%);
        color: #292929;
    }
    .stTextInput > div > input {
        border-radius: 10px;
        border: 2px solid #ffa751;
        background: #fffde9;
        font-size: 1.1rem;
        padding: 10px;
    }
    .stButton button {
        background-color: #ffa751;
        color: #fff;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.05em;
        transition: 0.2s;
    }
    .stMarkdown {
        background: rgba(255,255,255,0.85);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 14px;
        box-shadow: 0 2px 32px 0 #ffa75166;
    }
    </style>
""", unsafe_allow_html=True)


# ----- App UI -----
st.title("🌱 Grapevine Company Q&A")
st.caption("Ask anything about the company—get AI-driven, context-aware answers with style.")

rag = RAGSystem()

query = st.text_input("✨ What's your question?")

if query:
    with st.spinner("Thinking..."):
        answer = rag.answer_question(query)
    st.markdown(f"#### 🧠 Answer\n{answer}")

