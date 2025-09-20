import streamlit as st
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

# Initialize button states
if "run_pressed" not in st.session_state:
    st.session_state.run_pressed = False
if "confirm_pressed" not in st.session_state:
    st.session_state.confirm_pressed = False
if "refined_tasks" not in st.session_state:
    st.session_state.refined_tasks = []

# ------------------------
# UI
# ------------------------
st.title("🧠 Deep Researcher Agent")
st.write(
    "A local AI-powered research assistant that handles queries, reasoning, summarization, and exports reports."
)

# Upload documents
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

# Query input
query = st.text_input("Enter your research query:")

if st.button("Run Research") and query:
    st.session_state.run_pressed = True

# Break down query
if st.session_state.run_pressed and query:
    tasks = agent.reasoner.break_down_query(query)
    st.subheader("🔍 Subtasks Identified")
    refined_tasks = []
    for i, task in enumerate(tasks):
        new_task = st.text_input(f"Subtask {i+1}", value=task, key=f"task_{i}")
        if new_task.strip():
            refined_tasks.append(new_task.strip())

    st.session_state.refined_tasks = refined_tasks

    if st.button("Confirm & Process Tasks"):
        st.session_state.confirm_pressed = True

# Process tasks
if st.session_state.confirm_pressed and st.session_state.refined_tasks:
    refined_tasks = st.session_state.refined_tasks

    with st.spinner("Processing tasks, this may take a while..."):
        task_results = []
        for task in refined_tasks:
            q_emb = agent.embedding_engine.generate_embedding(task)
            docs = agent.vector_storage.retrieve_similar(q_emb, k=5)
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
                for d in docs[:5]:  # limit top 5 docs
                    st.write(f"**Doc ID:** {d['id']} | Score: {d['score']:.4f}")
                    st.write(d["content"])
                    st.write("---")
            else:
                st.write("⚠️ No relevant documents found.")

    st.subheader("📝 Final Report")
    st.text(final_report)

# Export
if "final_report" in st.session_state:
    st.subheader("📤 Export Results")

    # Generate PDF or Markdown in memory
    pdf_bytes = agent.export_report(st.session_state.final_report, format="pdf", return_bytes=True)
    md_bytes = agent.export_report(st.session_state.final_report, format="md", return_bytes=True)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="research_report.pdf",
            mime="application/pdf"
        )

    with col2:
        st.download_button(
            label="Download Markdown",
            data=md_bytes,
            file_name="research_report.md",
            mime="text/markdown"
        )
