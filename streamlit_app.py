import streamlit as st
from pathlib import Path
import pickle
import os
from research_agent.agent import ResearchAgent

# ------------------------
# Initialize Agent
# ------------------------
CACHE_PATH = "./cache/embeddings.pkl"

if "agent" not in st.session_state:
    st.session_state.agent = ResearchAgent(cache_path=CACHE_PATH)

agent = st.session_state.agent

st.title("🧠 Deep Researcher Agent")
st.write(
    "A local AI-powered research assistant that handles queries, reasoning, summarization, and exports reports."
)

# ------------------------
# Upload Documents
# ------------------------
uploaded_files = st.file_uploader(
    "Upload documents (TXT or JSON)", type=["txt", "json"], accept_multiple_files=True
)

if uploaded_files:
    docs = []
    for f in uploaded_files:
        content = f.read().decode("utf-8")
        docs.append({"id": f.name, "content": content, "metadata": {}})
    agent.add_documents(docs)
    st.success(f"{len(docs)} documents added to knowledge base.")

# ------------------------
# Query Input
# ------------------------
query = st.text_input("Enter your research query:")

if st.button("Run Research") and query:
    # Break down query into subtasks
    tasks = agent.reasoner.break_down_query(query)

    st.subheader("🔍 Subtasks Identified")
    refined_tasks = []
    for i, task in enumerate(tasks):
        new_task = st.text_input(f"Subtask {i+1}", value=task)
        if new_task.strip():
            refined_tasks.append(new_task.strip())

    if st.button("Confirm & Process Tasks"):
        explanation = agent.reasoner.explain_reasoning(refined_tasks)
        task_results = []

        for task in refined_tasks:
            st.markdown(f"**Processing subtask:** {task}")
            q_emb = agent.embedding_engine.generate_embedding(task)
            docs = agent.vector_storage.retrieve_similar(q_emb, k=5)

            # Display retrieved documents
            if docs:
                with st.expander(f"Retrieved docs for: {task}"):
                    for d in docs:
                        st.write(f"**Doc ID:** {d['id']} | **Score:** {d['score']:.4f}")
                        st.write(d["content"])
                        st.write("---")
            else:
                st.write("⚠️ No relevant documents found.")

            task_results.append({"task": task, "docs": docs})

        # Summarize all retrieved documents
        summary = agent.summarizer.summarize(
            [doc for r in task_results for doc in r["docs"]]
        )

        final_report = f"Query: {query}\n\nReasoning Steps:\n{explanation}\n\nSummary:\n{summary}"

        st.subheader("📝 Final Report")
        st.text(final_report)

        # Save for export
        st.session_state["final_report"] = final_report

# ------------------------
# Export
# ------------------------
if "final_report" in st.session_state:
    st.subheader("📤 Export Results")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export as PDF"):
            pdf_path = agent.export_report(st.session_state["final_report"], format="pdf")
            st.success(f"PDF saved at: {pdf_path}")

    with col2:
        if st.button("Export as Markdown"):
            md_path = agent.export_report(st.session_state["final_report"], format="md")
            st.success(f"Markdown saved at: {md_path}")
