import os
from .embedding import LocalEmbeddingEngine
from .storage import VectorStorage
from .reasoning import MultiStepReasoner
from .summarizer import ExtractiveSummarizer
from .agent import ResearchAgent
from fpdf import FPDF

class ResearchAgent:
    def __init__(self, cache_path="./cache/embeddings.pkl"):
        self.embedding_engine = LocalEmbeddingEngine()
        self.vector_storage = VectorStorage(cache_path)
        self.reasoner = MultiStepReasoner()
        self.summarizer = ExtractiveSummarizer()

    def add_documents(self, docs):
        self.vector_storage.add_documents(docs, self.embedding_engine)

    def export_report(self, text, format="pdf"):
        os.makedirs("outputs", exist_ok=True)
        if format == "pdf":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, text)
            path = f"outputs/report.pdf"
            pdf.output(path)
            return path
        elif format == "md":
            path = f"outputs/report.md"
            with open(path, "w") as f:
                f.write(text)
            return path

