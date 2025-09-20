import os
import io
from .embedding import LocalEmbeddingEngine
from .storage import VectorStorage
from .reasoning import MultiStepReasoner
from .summarizer import ExtractiveSummarizer
from fpdf import FPDF

class ResearchAgent:
    def __init__(self, cache_path="./cache/embeddings.pkl"):
        self.embedding_engine = LocalEmbeddingEngine()
        self.vector_storage = VectorStorage(cache_path)
        self.reasoner = MultiStepReasoner()
        self.summarizer = ExtractiveSummarizer()

    def add_documents(self, docs):
        self.vector_storage.add_documents(docs, self.embedding_engine)

    def export_report(self, report_text, format="pdf", return_bytes=False):
        """
        Export the report as PDF or Markdown.
        If return_bytes=True, returns the file content as bytes.
        """
        if format == "pdf":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            for line in report_text.split("\n"):
                pdf.multi_cell(0, 5, line)
            
            if return_bytes:
                # Get PDF content as bytes
                pdf_bytes = pdf.output(dest='S').encode('latin1')
                return pdf_bytes
            else:
                path = "./report.pdf"
                pdf.output(path)
                return path
    
        elif format == "md":
            md_content = report_text
            if return_bytes:
                return md_content.encode("utf-8")
            else:
                path = "./report.md"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(md_content)
                return path
