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

agent = st.session_state.agent

# ------------------------
# Initialize Chat History
# ------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------
# UI
# ------------------------
st.title("🧠 Deep Researcher Agent")
st.write(
    "A local AI-powered research assistant that handles queries, reasoning, summarization, and exports reports."
)

# ------------------------
# Upload Documents
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
        agent.vector_storage.add_documents(docs, agent.embedding_engine)
        st.success(f"{len(docs)} documents added to knowledge base.")

# ------------------------
# Show Chat History
# ------------------------
if st.session_state.chat_history:
    st.subheader("💬 Conversation History")
    
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")
            
            # Display analysis if available
            if "analysis" in msg and msg["analysis"]:
                analysis_md = "<details><summary>📊 Analysis</summary>"
                analysis_md += "<table border='1' style='border-collapse: collapse; text-align: left;'>"
                analysis_md += "<tr><th>Document ID</th><th>Similarity</th><th>Top-K</th></tr>"
                
                seen_docs = set()
                for doc in msg["analysis"]:
                    if doc['id'] in seen_docs:
                        continue
                    similarity = f"{doc['score']:.4f}"
                    chosen_mark = "✅" if doc.get("chosen") else ""
                    analysis_md += f"<tr><td>{doc['id']}</td><td>{similarity}</td><td>{chosen_mark}</td></tr>"
                    seen_docs.add(doc['id'])
                
                analysis_md += "</table></details>"
                st.markdown(analysis_md, unsafe_allow_html=True)

# ------------------------
# Chat Input & Export Buttons
# ------------------------
st.subheader("💬 Ask a Question / Export Full Session")

user_input = st.text_input("Enter your query here:", key="chat_input")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if user_input.strip() and st.button("Send", key="send_button"):
        user_msg = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        with st.spinner("Generating response..."):
            context = [
                (msg["content"], msg.get("response", ""))
                for msg in st.session_state.chat_history
                if msg["role"] == "user"
            ]
            response, analysis = agent.process_query(user_msg, context=context, top_k=3)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "analysis": analysis,
            "query": user_msg
        })

# Export buttons
if st.session_state.chat_history:
    with col2:
        pdf_bytes = agent.export_report(st.session_state.chat_history, format="pdf", return_bytes=True)
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="research_session.pdf",
            mime="application/pdf"
        )

    with col3:
        md_bytes = agent.export_report(st.session_state.chat_history, format="md", return_bytes=True)
        st.download_button(
            label="Download Markdown",
            data=md_bytes,
            file_name="research_session.md",
            mime="text/markdown"
        )
