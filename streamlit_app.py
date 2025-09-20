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
        agent.add_documents(docs)
        st.success(f"{len(docs)} documents added to knowledge base.")

# ------------------------
# Chat Input
# ------------------------
if st.button("Send") and user_input.strip():
    user_msg = user_input.strip()
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    
    # Generate AI response with analysis
    with st.spinner("Generating response..."):
        context = [(msg["content"], msg.get("response","")) 
                   for msg in st.session_state.chat_history 
                   if msg["role"] == "user"]
        
        response, analysis = agent.process_query(user_msg, context=context, top_k=3)
        
        # Add AI message + analysis to chat
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "analysis": analysis
        })

# ------------------------
# Display Chat History
# ------------------------
st.subheader("💬 Conversation")
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        assistant_msg = st.chat_message("assistant")
        assistant_msg.markdown(msg["content"])
        
        # Show contributing documents
        if "analysis" in msg:
            with assistant_msg.expander("📄 Contributing Documents"):
                for doc in msg["analysis"]:
                    st.write(f"- **{doc['id']}** | Similarity: {doc['score']:.4f}")

# ------------------------
# Export Full Session
# ------------------------
if st.session_state.chat_history:
    full_report = ""
    for msg in st.session_state.chat_history:
        full_report += f"{msg['role'].capitalize()}: {msg['content']}\n\n"

    st.subheader("📤 Export Full Session")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download PDF",
            data=agent.export_report(full_report, format="pdf", return_bytes=True),
            file_name="research_session.pdf",
            mime="application/pdf"
        )

    with col2:
        st.download_button(
            label="Download Markdown",
            data=agent.export_report(full_report, format="md", return_bytes=True),
            file_name="research_session.md",
            mime="text/markdown"
        )
