import streamlit as st
from pathlib import Path
from research_agent.agent import ResearchAgent

# ------------------------
# Constants
# ------------------------
CACHE_PATH = "./cache/embeddings.pkl"

# ------------------------
# Load Agent (cached)
# ------------------------
@st.cache_resource(show_spinner=False)
def load_agent(cache_path=CACHE_PATH):
    return ResearchAgent(cache_path=cache_path)

agent = load_agent()

st.title("🧠 Deep Researcher Agent")
st.write(
    "A local AI-powered research assistant that handles queries, reasoning, summarization, and exports reports."
)

# ------------------------
# Upload Documents (cached)
# ------------------------
@st.cache_data(show_spinner=False)
def add_documents(_agent, uploaded_files):
    docs = []
    for f in uploaded_files:
        content = f.read().decode("utf-8")
        docs.append({"id": f.name, "content": content, "metadata": {}})
    _agent.add_documents(docs)
    return len(docs)

uploaded_files = st.file_uploader(
    "Upload documents (TXT or JSON)", type=["txt", "json"], accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Adding documents..."):
        num_docs = add_documents(agent, uploaded_files)
    st.success(f"{num_docs} documents added to knowledge base.")

# ------------------------
# Query Input
# ------------------------
query = st.text_input("Enter your research query:")

if st.button("Run Research") and query:
    with st.spinner("Breaking down query into subtasks..."):
        tasks = agent.reasoner.break_down_query(query)

    st.subheader("🔍 Subtasks Identified")
    refined_tasks = []
    for i, task in enumerate(tasks):
        new_task = st.text_input(f"Subtask {i+1}", value=task)
        if new_task.strip():
            refined_tasks.append(new_task.strip())

    if st.button("Confirm & Process Tasks") and refined_tasks:
        task_results = []

        explanation = agent.reasoner.explain_reasoning(refined_tasks)
        st.subheader("⚡ Processing Subtasks")

        for task in refined_tasks:
            with st.spinner(f"Processing: {task}"):
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
                    st.write(f"⚠️ No relevant documents found for '{task}'.")

                task_results.append({"task": task, "docs": docs})

        # Summarize results
        with st.spinner("Generating summary..."):
            summary = agent.summarizer.summarize(
                [doc for r in task_results for doc in r["docs"]]
            )

        final_report = f"Query: {query}\n\nReasoning Steps:\n{explanation}\n\nSummary:\n{summary}"

        st.subheader("📝 Final Report")
        st.text_area("Final Report", final_report, height=300)

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
            with st.spinner("Generating PDF..."):
                pdf_path = agent.export_report(st.session_state["final_report"], format="pdf")
            st.success(f"PDF saved at: {pdf_path}")

    with col2:
        if st.button("Export as Markdown"):
            with st.spinner("Generating Markdown..."):
                md_path = agent.export_report(st.session_state["final_report"], format="md")
            st.success(f"Markdown saved at: {md_path}")
