import os
import streamlit as st
import PyPDF2
from pathlib import Path
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

agent = st.session_state.agent

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # stores tuples: (query, response)

# ------------------------
# UI
# ------------------------
st.title("🧠 Deep Researcher Agent")
st.write(
    "A local AI-powered research assistant. Ask research questions, explore follow-ups, and export the full session."
)

# Upload documents
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
                pdf_reader = PyPDF2.PdfReader(f)
                content = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content += page_text + "\n"
            else:
                content = f.read().decode("utf-8")
            
            docs.append({"id": f.name, "content": content, "metadata": {}})
        except Exception as e:
            st.warning(f"Failed to process {f.name}: {e}")

    if docs:
        agent.add_documents(docs)
        st.success(f"{len(docs)} documents added to knowledge base.")

# ------------------------
# Chat input
# ------------------------
st.subheader("💬 Research Chat")
user_query = st.text_input("Enter your question or follow-up:")

if st.button("Send") and user_query.strip():
    with st.spinner("Processing..."):
        q_emb = agent.embedding_engine.generate_embedding(user_query)
        docs = agent.vector_storage.retrieve_similar(q_emb, k=5)

        summary = agent.summarizer.summarize(docs)
        explanation = agent.reasoner.explain_reasoning([user_query])

        response = f"**Reasoning Steps:**\n{explanation}\n\n**Summary:**\n{summary}"
        st.session_state.chat_history.append((user_query, response))

# ------------------------
# Show chat history
# ------------------------
if st.session_state.chat_history:
    for i, (q, r) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**🧑 Query {i}:** {q}")
        st.markdown(f"**🤖 Response {i}:**\n{r}")
        st.write("---")

# ------------------------
# Export results
# ------------------------
if st.session_state.chat_history:
    combined_report = ""
    for i, (q, r) in enumerate(st.session_state.chat_history, 1):
        combined_report += f"Query {i}: {q}\n\nResponse {i}:\n{r}\n\n---\n\n"

    st.subheader("📤 Export Results")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download PDF",
            data=agent.export_report(
                combined_report,
                format="pdf",
                return_bytes=True
            ),
            file_name="research_session.pdf",
            mime="application/pdf"
        )

    with col2:
        st.download_button(
            label="Download Markdown",
            data=agent.export_report(
                combined_report,
                format="md",
                return_bytes=True
            ),
            file_name="research_session.md",
            mime="text/markdown"
        )
