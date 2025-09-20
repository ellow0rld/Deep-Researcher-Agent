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
if "run_pressed" not in st.session_state:
    st.session_state.run_pressed = False
if "confirm_pressed" not in st.session_state:
    st.session_state.confirm_pressed = False
if "refined_tasks" not in st.session_state:
    st.session_state.refined_tasks = []
if "final_report" not in st.session_state:
    st.session_state.final_report = None

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

if st.button("Run Research") and query:
    st.session_state.run_pressed = True
    st.session_state.confirm_pressed = False
    st.session_state.final_report = None
    # Generate subtasks
    st.session_state.refined_tasks = agent.reasoner.break_down_query(query)

# ------------------------
# Subtask refinement UI
# ------------------------
if st.session_state.run_pressed and query:
    st.subheader("🔍 Subtasks Identified")

    tasks = st.session_state.refined_tasks

    # Editable list of tasks
    new_tasks = []
    remove_indices = []
    for i, task in enumerate(tasks):
        cols = st.columns([8, 2])
        new_task = cols[0].text_input(f"Subtask {i+1}", value=task, key=f"task_{i}")
        if cols[1].button("❌", key=f"remove_{i}"):
            remove_indices.append(i)
        else:
            if new_task.strip():
                new_tasks.append(new_task.strip())

    # Remove marked tasks
    for idx in sorted(remove_indices, reverse=True):
        if idx < len(new_tasks):
            new_tasks.pop(idx)

    # Add new sub-task manually
    if st.button("➕ Add new sub-task"):
        new_tasks.append("")

    st.session_state.refined_tasks = new_tasks

    if st.button("Confirm & Process Tasks"):
        st.session_state.confirm_pressed = True

# ------------------------
# Process tasks
# ------------------------
if st.session_state.confirm_pressed and st.session_state.refined_tasks:
    refined_tasks = st.session_state.refined_tasks

    with st.spinner("Processing tasks, this may take a while..."):
        task_results = []
        for task in refined_tasks:
            q_emb = agent.embedding_engine.generate_embedding(task)
            docs = agent.vector_storage.retrieve_similar(q_emb, k=len(docs))
            task_results.append({"task": task, "docs": docs})

        # Summarize all retrieved documents
        summary = agent.summarizer.summarize(
            [doc for r in task_results for doc in r["docs"]]
        )
        explanation = agent.reasoner.explain_reasoning(refined_tasks)
        final_report = f"Query: {query}\n\nReasoning Steps:\n{explanation}\n\nSummary:\n{summary}"
        st.session_state.final_report = final_report

    # Display results
    st.subheader("📝 Task Results")
    for r in task_results:
        task = r["task"]
        docs = r["docs"]
        with st.expander(f"Results for: {task}"):
            if docs:
                for d in docs:
                    st.write(f"**Doc ID:** {d['id']} | Score: {d['score']:.4f}")
                    st.write(d["content"][:500] + "...")
                    st.write("---")
            else:
                st.write("⚠️ No relevant documents found.")

    st.subheader("📝 Final Report")
    st.text(st.session_state.final_report)

# ------------------------
# Export
# ------------------------
if st.session_state.final_report:
    st.subheader("📤 Export Results")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download PDF",
            data=agent.export_report(
                st.session_state["final_report"], 
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
                st.session_state["final_report"], 
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
        new_tasks = agent.reasoner.break_down_query(follow_up)
        st.session_state.refined_tasks.extend(new_tasks)
        st.session_state.run_pressed = True
        st.session_state.confirm_pressed = False
