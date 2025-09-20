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

    def add_documents(self, docs):
        self.vector_storage.add_documents(docs, self.embedding_engine)

    def export_report(self, chat_history, format="pdf", return_bytes=False):
    """
    Export full chat session with similarity analysis.
    """
    report_text = ""
    for msg in chat_history:
        if msg["role"] == "user":
            report_text += f"Q: {msg['content']}\n"
        else:
            report_text += f"A: {msg['content']}\n"
            if "analysis" in msg:
                report_text += "Contributing Documents:\n"
                for doc in msg["analysis"]:
                    report_text += f"- {doc['doc_id']} | Similarity: {doc['similarity']:.4f}\n"
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
            
    def process_query(self, query, context=None, top_k=5):
        """
        Process a query and return a meaningful response along with top contributing documents.
        Returns:
          - response: string
          - analysis: list of dicts with doc id and similarity
        """
        # Generate query embedding
        q_emb = self.embedding_engine.generate_embedding(query)
        
        # Retrieve top-k similar documents
        docs = self.vector_storage.retrieve_similar(q_emb, k=top_k)
        
        # For now: simple combined answer (concatenate top docs' first 300 chars)
        response = "\n".join([f"{d['content'][:300]}..." for d in docs])
        
        # Include doc IDs and similarity scores for analysis
        analysis = [{"doc_id": d["id"], "similarity": d["score"]} for d in docs]
        
        return response, analysis
