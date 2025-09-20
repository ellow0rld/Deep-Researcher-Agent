import os
from fpdf import FPDF

class Exporter:
    def export_pdf(self, text, filename="report.pdf"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in text.split("\n"):
            pdf.multi_cell(0, 5, line)
        pdf.output(filename)
        return os.path.abspath(filename)

    def export_md(self, text, filename="report.md"):
        with open(filename, "w") as f:
            f.write(text)
        return os.path.abspath(filename)
