import io
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from research_agent.embedding import LocalEmbeddingEngine
from research_agent.storage import VectorStorage
from research_agent.reasoning import MultiStepReasoner
from research_agent.summarizer import ExtractiveSummarizer
from fpdf import FPDF

class ResearchAgent:
    def __init__(self, cache_path="./cache/embeddings.pkl"):
        self.embedding_engine = LocalEmbeddingEngine()
        self.vector_storage = VectorStorage(cache_path)
        self.reasoner = MultiStepReasoner()
        self.summarizer = ExtractiveSummarizer()

    def add_documents(self, docs, embedding_engine):
        existing_ids = {v["id"] for v in self.vectors}
        for doc in docs:
            if doc["id"] not in existing_ids:
                emb = embedding_engine.generate_embedding(doc["content"])
                self.vectors.append({"id": doc["id"], "content": doc["content"], "embedding": emb, "metadata": doc.get("metadata", {})})

    def export_report(self, chat_history, format="pdf", return_bytes=False):
        """
        Export full chat session with analysis.
        Uses default fonts, safe for ASCII characters.
        """
        report_text = ""
        for msg in chat_history:
            if msg["role"] == "user":
                report_text += f"Q: {msg['content']}\n"
            else:
                report_text += f"A: {msg['content']}\n"
                if "analysis" in msg:
                    report_text += "Contributing Documents:\n"
                    vis = []
                    for doc in msg["analysis"]:
                        chosen_mark = "*" if doc.get("chosen") else ""
                        if doc['id'] not in vis:
                            report_text += f"- {doc['id']} | Similarity: {doc['score']:.4f} {chosen_mark}\n"
                            vis.append(doc['id'])
            report_text += "\n"
            safe_text = report_text.encode("ascii", errors="ignore").decode("ascii")
    
        if format == "pdf":
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            for line in safe_text.split("\n"):
                pdf.multi_cell(0, 5, line)
            if return_bytes:
                return pdf.output(dest='S').encode('latin1')
            else:
                path = "./full_report.pdf"
                pdf.output(path)
                return path
        else:
            if return_bytes:
                return report_text.encode("utf-8")
            else:
                path = "./full_report.md"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(report_text)
                return path

    def process_query(self, query, context=None, top_k=1):
        """
        Process a query into subtasks and answer each separately.
        Returns:
          - response (markdown string with reasoning + answers)
          - analysis (all doc scores)
        """
        # Step 1: Break query into subtasks
        tasks = self.reasoner.break_down_query(query)
    
        # Step 2: For each subtask → retrieve docs → generate answer
        subtask_responses = []
        all_docs = []
        for i, task in enumerate(tasks, 1):
            q_emb = self.embedding_engine.generate_embedding(task)
            docs = self.vector_storage.retrieve_all_with_scores(q_emb)
    
            top_docs = docs[:top_k]
            answer_text = self.reasoner.answer_query(task, top_docs)
    
            subtask_responses.append(
                f"**Subtask {i}: {task}**\n\n{answer_text}\n"
            )
            all_docs.extend(top_docs)
    
        # Step 3: Merge into final markdown
        response = "### Reasoning Steps\n"
        for i, task in enumerate(tasks, 1):
            response += f"{i}. {task}\n"
        response += "\n### Answers\n" + "\n".join(subtask_responses)
    
        # Step 4: Collect analysis (unique doc IDs with scores)
        analysis = []
        seen = set()
        for d in all_docs:
            if d["id"] not in seen:
                analysis.append({
                    "id": d["id"],
                    "score": d["score"],
                    "chosen": True
                })
                seen.add(d["id"])
    
        return response, analysis

