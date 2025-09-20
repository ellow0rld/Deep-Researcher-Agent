# research_agent/summarizer.py
class ExtractiveSummarizer:
    def summarize(self, docs):
        # Simple extractive summarization: return first sentence of each doc
        summary = ""
        for doc in docs:
            first_sentence = doc["content"].split(".")[0]
            summary += f"- {first_sentence}.\n"
        return summary if summary else "No content to summarize."
