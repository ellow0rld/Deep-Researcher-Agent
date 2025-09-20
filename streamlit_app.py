import os
import streamlit as st
import PyPDF2
from research_agent.agent import ResearchAgent

# ------------------------
# Constants
# ------------------------
CACHE_PATH = "./cache/embeddings.pkl"

# ------------------------
# Initialize Agent
# ------------------------
if "agent" not in st.session_state:
    with st.spinner("Loading AI model... this may take a while"):
        st.session_state.agent = ResearchAgent(cache_path=CACHE_PATH)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{role: "user"/"assistant", content: str}, ...]

agent = st.session_state.agent

# ------------------------
# UI
# ------------------------
st.title("🧠 Deep Researcher Agent (Chat Mode)")
st.write("A conversational AI-powered research assistant. Upload docs, ask questions, and refine with follow-ups.")

# ------------------------
# Document Upload
# ------------------------
uploaded_files = st.file_uploader(
    "Upload documents (TXT, JSON, or PDF)",
    type=["txt", "json", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    docs = []
    for f in uploaded_files:
        try:
            if f.name.endswith(".pdf"):
                reader = PyPDF2.PdfReader(f)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            else:
                text = f.read().decode("utf-8")

            docs.append({"id": f.name, "content": text, "metadata": {}})
        except Exception as e:
            st.warning(f"Could not process {f.name}: {e}")

    if docs:
        agent.add_documents(docs)
        st.success(f"✅ {len(docs)} documents added.")

# ------------------------
# Chat Input
# ------------------------
query = st.chat_input("Type your research query or follow-up...")

if query:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Build context from all history
    context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history]
    )

    # Agent generates response
    with st.spinner("Thinking..."):
        explanation = agent.reasoner.explain_reasoning([query])
        q_emb = agent.embedding_engine.generate_embedding(query)
        docs = agent.vector_storage.retrieve_similar(q_emb, k=5)
        summary = agent.summarizer.summarize(docs)

        response = f"**Reasoning:**\n{explanation}\n\n**Summary:**\n{summary}"

    # Add assistant message
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# ------------------------
# Display Chat History
# ------------------------
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# ------------------------
# Export Results
# ------------------------
if st.session_state.chat_history:
    st.subheader("📤 Export Full Session")
    full_session_text = "\n\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download PDF",
            data=agent.export_report(full_session_text, format="pdf", return_bytes=True),
            file_name="research_session.pdf",
            mime="application/pdf"
        )

    with col2:
        st.download_button(
            label="Download Markdown",
            data=agent.export_report(full_session_text, format="md", return_bytes=True),
            file_name="research_session.md",
            mime="text/markdown"
        )
