import os
import streamlit as st
import PyPDF2
from research_agent.agent import ResearchAgent

CACHE_PATH = "./cache/embeddings.pkl"

if "agent" not in st.session_state:
    with st.spinner("Loading AI model... this may take a while"):
        st.session_state.agent = ResearchAgent(cache_path=CACHE_PATH)

agent = st.session_state.agent

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Deep Researcher Agent")
st.write("A local AI-powered research assistant that handles queries, reasoning, summarization, and exports reports.")

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

st.subheader("💬 Conversation")
chat_container = st.container()

user_input = st.chat_input("Enter your query here:")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Generating response..."):
        context = [(msg["content"], msg.get("response", "")) for msg in st.session_state.chat_history if msg["role"] == "user"]
        response, analysis = agent.process_query(user_input, context=context, top_k=1)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "analysis": analysis,
        "query": user_input
    })

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        assistant_msg = st.chat_message("assistant")
        assistant_msg.subheader("Final Answer")
        assistant_msg.markdown(msg["content"]) 

        if "analysis" in msg and msg["analysis"]:
            with assistant_msg.expander("Analysis"):
                for doc in msg["analysis"]:
                    chosen_mark = "*" if doc.get("chosen") else ""
                    st.markdown(f"- {doc['id']} | Similarity: {doc['score']:.4f} {chosen_mark}")

user_input = st.chat_input("Enter your query here:", key="main_chat_input")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Generating response..."):
        context = [(msg["content"], msg.get("response", "")) for msg in st.session_state.chat_history if msg["role"] == "user"]
        response, analysis = agent.process_query(user_input, context=context, top_k=1)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "analysis": analysis,
        "query": user_input
    })
    st.experimental_rerun()
    

if st.session_state.chat_history: 
    col1, col2 = st.columns(2) 
    with col1: 
        st.download_button( 
            label="Download PDF", 
            data=agent.export_report(st.session_state.chat_history, format="pdf", return_bytes=True), 
            file_name="research_session.pdf", mime="application/pdf" ) 
    with col2: 
        st.download_button( 
            label="Download Markdown", 
            data=agent.export_report(st.session_state.chat_history, format="md", return_bytes=True), 
            file_name="research_session.md", mime="text/markdown" )
