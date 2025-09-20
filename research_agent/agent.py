import os
import io
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

    def export_report(self, report_text, format="pdf", return_bytes=False):
        if format == "pdf":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            for line in report_text.split("\n"):
                pdf.multi_cell(0, 5, line)
    
            if return_bytes:
                return pdf.output(dest='S').encode('latin1')  # PDF bytes
            else:
                path = "./report.pdf"
                pdf.output(path)
                return path
    
        elif format == "md":
            if return_bytes:
                return report_text.encode("utf-8")
            else:
                path = "./report.md"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(report_text)
                return path
    
    
