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
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "all_reports" not in st.session_state:
    st.session_state.all_reports = []

# ------------------------
# UI
# ------------------------
st.title("🧠 Deep Researcher Agent")
st.write(
    "A local AI-powered research assistant that handles queries, reasoning, summarization, and exports reports."
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
                # For TXT and JSON
                content = f.read().decode("utf-8")
            
            docs.append({"id": f.name, "content": content, "metadata": {}})
        except Exception as e:
            st.warning(f"Failed to process {f.name}: {e}")

    if docs:
        agent.add_documents(docs)
        st.success(f"{len(docs)} documents added to knowledge base.")

# ------------------------
# Query input
# ------------------------
query = st.text_input("Enter your research query:")

if st.button("Run Research") and query.strip():
    with st.spinner("Processing query..."):
        q_emb = agent.embedding_engine.generate_embedding(query)
        docs = agent.vector_storage.retrieve_similar(q_emb, k=5)

        summary = agent.summarizer.summarize(docs)
        explanation = agent.reasoner.explain_reasoning([query])

        report = f"Query: {query}\n\nReasoning Steps:\n{explanation}\n\nSummary:\n{summary}"
        st.session_state.final_report = report
        st.session_state.all_reports.append(report)

# ------------------------
# Show results
# ------------------------
if st.session_state.final_report:
    st.subheader("📝 Latest Report")
    st.text(st.session_state.final_report)

    st.subheader("📚 Full Research Session")
    for i, rep in enumerate(st.session_state.all_reports, 1):
        with st.expander(f"Report {i}"):
            st.text(rep)

# ------------------------
# Export
# ------------------------
if st.session_state.all_reports:
    combined_report = "\n\n---\n\n".join(st.session_state.all_reports)

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
            file_name="research_report.pdf",
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
            file_name="research_report.md",
            mime="text/markdown"
        )

# ------------------------
# Follow-up queries
# ------------------------
if st.session_state.final_report:
    st.subheader("🔄 Follow-up Query")
    follow_up = st.text_input("Ask a follow-up question:")
    if st.button("Run Follow-up") and follow_up.strip():
        with st.spinner("Processing follow-up..."):
            q_emb = agent.embedding_engine.generate_embedding(follow_up)
            docs = agent.vector_storage.retrieve_similar(q_emb, k=5)

            summary = agent.summarizer.summarize(docs)
            explanation = agent.reasoner.explain_reasoning([follow_up])

            report = f"Follow-up Query: {follow_up}\n\nReasoning Steps:\n{explanation}\n\nSummary:\n{summary}"
            st.session_state.final_report = report
            st.session_state.all_reports.append(report)
