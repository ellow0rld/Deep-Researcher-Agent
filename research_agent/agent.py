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
    
        if format == "pdf":
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            for line in report_text.split("\n"):
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

    def process_query(self, query, context=None, top_k=3):
        """
        Process a query and return a meaningful response along with all documents' similarity scores.
        Returns:
          - response: string
          - analysis: list of dicts with doc id, similarity, and top-k flag
        """
        # Generate query embedding
        q_emb = self.embedding_engine.generate_embedding(query)

        # Retrieve all documents' similarity scores
        all_docs = self.vector_storage.retrieve_similar(q_emb, k=len(self.vector_storage.vectors))

        # Pick top-k documents
        top_docs = all_docs[:top_k]

        # Simple response
        response = "\n".join([f"{d['content']}..." for d in top_docs])

        # Prepare analysis for all documents
        analysis = []
        top_doc_ids = [d['id'] for d in top_docs]
        for d in all_docs:
            analysis.append({
                "id": d["id"],
                "score": d["score"],
                "chosen": d["id"] in top_doc_ids
            })

        return response, analysis






