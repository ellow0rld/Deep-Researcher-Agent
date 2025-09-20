import streamlit as st
import time
from research_agent import ResearchAgent

st.title("🧠 Deep Researcher Agent Debug")

print("Streamlit app started")

# ------------------------
# Initialize agent with debug logs
# ------------------------
try:
    if "agent" not in st.session_state:
        print("Initializing ResearchAgent...")
        st.session_state.agent = ResearchAgent(model_path="./models/all-MiniLM-L6-v2")
        print("ResearchAgent initialized")
    agent = st.session_state.agent
except Exception as e:
    st.error(f"Error initializing agent: {e}")
    print(f"Error initializing agent: {e}")
    agent = None

# ------------------------
# Upload Documents
# ------------------------
uploaded_files = st.file_uploader("Upload documents (TXT or JSON)", type=["txt", "json"], accept_multiple_files=True)
if uploaded_files and agent:
    try:
        docs = []
        for f in uploaded_files:
            content = f.read().decode("utf-8")
            docs.append({"id": f.name, "content": content, "metadata": {}})
        agent.add_documents(docs)
        st.success(f"{len(docs)} documents added to knowledge base.")
        print(f"{len(docs)} documents added")
    except Exception as e:
        st.error(f"Error adding documents: {e}")
        print(f"Error adding documents: {e}")

# ------------------------
# Query Input
# ------------------------
query = st.text_input("Enter your research query:")

if st.button("Run Research") and query and agent:
    try:
        print(f"Running query: {query}")
        tasks = agent.reasoner.break_down_query(query)
        st.subheader("🔍 Subtasks identified")
        refined_tasks = []

        for i, task in enumerate(tasks):
            new_task = st.text_input(f"Subtask {i+1}", value=task)
            if new_task.strip():
                refined_tasks.append(new_task.strip())
        print(f"Refined tasks: {refined_tasks}")

        if st.button("Confirm & Process Tasks"):
            explanation = agent.reasoner.explain_reasoning(refined_tasks)
            task_results = []

            for task in refined_tasks:
                st.markdown(f"**Processing subtask:** {task}")
                print(f"Generating embedding for task: {task}")
                q_emb = agent.embedding_engine.generate_embedding(task)
                docs = agent.vector_storage.retrieve_similar(q_emb, k=5)
                print(f"Retrieved {len(docs)} docs")

                if docs:
                    with st.expander(f"Retrieved docs for: {task}"):
                        for d in docs:
                            st.write(f"**Doc ID:** {d['id']} | **Score:** {d['score']:.4f}")
                            st.write(d["content"])
                            st.write("---")
                else:
                    st.write("⚠️ No relevant documents found.")

                task_results.append({"task": task, "docs": docs})

            summary = agent.summarizer.summarize([doc for r in task_results for doc in r["docs"]])
            final_report = f"Query: {query}\n\nReasoning Steps:\n{explanation}\n\nSummary:\n{summary}"
            st.subheader("📝 Final Report")
            st.text(final_report)
            st.session_state["final_report"] = final_report
            print("Final report generated")

    except Exception as e:
        st.error(f"Error during query processing: {e}")
        print(f"Error during query processing: {e}")

# ------------------------
# Export
# ------------------------
if "final_report" in st.session_state and agent:
    st.subheader("📤 Export Results")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export as PDF"):
            try:
                pdf_path = agent.export_report(st.session_state["final_report"], format="pdf")
                st.success(f"PDF saved at: {pdf_path}")
                print(f"PDF exported to: {pdf_path}")
            except Exception as e:
                st.error(f"Error exporting PDF: {e}")
                print(f"Error exporting PDF: {e}")

    with col2:
        if st.button("Export as Markdown"):
            try:
                md_path = agent.export_report(st.session_state["final_report"], format="md")
                st.success(f"Markdown saved at: {md_path}")
                print(f"Markdown exported to: {md_path}")
            except Exception as e:
                st.error(f"Error exporting Markdown: {e}")
                print(f"Error exporting Markdown: {e}")
