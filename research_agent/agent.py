import io
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from research_agent.embedding import LocalEmbeddingEngine
from research_agent.storage import VectorStorage
from research_agent.reasoning import MultiStepReasoner
from research_agent.summarizer import ExtractiveSummarizer
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("DejaVu", "B", 14)
        self.cell(0, 10, "Deep Researcher Agent Report", ln=True, align="C")

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
        report_text = ""
        for msg in chat_history:
            if msg["role"] == "user":
                report_text += f"Q: {msg['content']}\n"
            else:
                report_text += f"A: {msg['content']}\n"
                if "analysis" in msg:
                    report_text += "Contributing Documents:\n"
                    for doc in msg["analysis"]:
                        chosen_mark = "✅" if doc.get("chosen") else ""
                        report_text += f"- {doc['id']} | Similarity: {doc['score']:.4f} {chosen_mark}\n"
            report_text += "\n"
    
        pdf = PDF()
        pdf.add_page()
        # Add a TTF font that supports Unicode
        pdf.add_font("DejaVu", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)
    
        for line in report_text.split("\n"):
            pdf.multi_cell(0, 5, line)
    
        if return_bytes:
            return pdf.output(dest='S').encode('latin1', errors='replace')  # now safe
        else:
            path = "./full_report.pdf"
            pdf.output(path)
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

        # Simple response (concatenate first 300 chars of top-k docs)
        response = "\n".join([f"{d['content'][:300]}..." for d in top_docs])

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



