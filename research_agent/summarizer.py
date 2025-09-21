# research_agent/summarizer.py
class ExtractiveSummarizer:
    def summarize(self, docs):
        # Simple extractive summarization: return first sentence of each doc
        summary = ""
        for doc in docs:
            content = doc["content"].split(".")
            summary += f"- {content}.\n"
        return summary if summary else "No content to summarize."

