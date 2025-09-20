from .embedding import LocalEmbeddingEngine
from .storage import VectorStorage
from .reasoning import MultiStepReasoner
from .summarizer import ExtractiveSummarizer
from .exporter import Exporter

class ResearchAgent:
    """
    Research Agent with interactive query refinement and reasoning explanations.
    """
    def __init__(self, model_path="./models/all-MiniLM-L6-v2", storage_path="./research_storage"):
        self.embedding_engine = LocalEmbeddingEngine(model_path)
        self.vector_storage = VectorStorage(storage_path)
        self.reasoner = MultiStepReasoner()
        self.summarizer = ExtractiveSummarizer()
        self.exporter = Exporter()

    def add_documents(self, documents):
        for doc in documents:
            emb = self.embedding_engine.generate_embedding(doc["content"])
            self.vector_storage.store_document(doc["id"], doc["content"], doc.get("metadata", {}), emb)

    def process_query(self, query):
        # 1️⃣ Break query into subtasks
        tasks = self.reasoner.break_down_query(query)
        explanation = self.reasoner.explain_reasoning(tasks)

        # 2️⃣ Interactive refinement
        refined_tasks = []
        print("\nInteractive Query Refinement:")
        for task in tasks:
            print(f"\nSubtask suggested: '{task}'")
            user_input = input("Press Enter to accept, type modification, or 'skip' to remove: ").strip()
            if user_input.lower() == "skip":
                continue
            elif user_input:
                refined_tasks.append(user_input)
            else:
                refined_tasks.append(task)
        if not refined_tasks:
            refined_tasks = tasks  # fallback if all skipped

        # 3️⃣ Process each subtask
        task_results = []
        for task in refined_tasks:
            q_emb = self.embedding_engine.generate_embedding(task)
            docs = self.vector_storage.retrieve_similar(q_emb, k=5)
            task_results.append({"task": task, "docs": docs})

        # 4️⃣ Generate detailed reasoning explanation
        reasoning_details = "Detailed Reasoning Steps:\n"
        for idx, r in enumerate(task_results, 1):
            reasoning_details += f"\nSubtask {idx}: {r['task']}\n"
            if r['docs']:
                reasoning_details += "Relevant documents considered:\n"
                for doc in r['docs']:
                    reasoning_details += f"- ({doc['score']:.3f}) {doc['content'][:120]}...\n"
            else:
                reasoning_details += "No relevant documents found.\n"

        # 5️⃣ Summarize all retrieved information
        all_docs = [doc for r in task_results for doc in r["docs"]]
        summary = self.summarizer.summarize(all_docs)

        # 6️⃣ Compose final report
        final_report = f"Query: {query}\n\nInitial Reasoning:\n{explanation}\n\n{reasoning_details}\nSummary:\n{summary}"
        return final_report, task_results

    def export_report(self, text, format="pdf"):
        if format == "pdf":
            return self.exporter.export_pdf(text)
        elif format == "md":
            return self.exporter.export_md(text)
        else:
            raise ValueError("Format must be 'pdf' or 'md'")
