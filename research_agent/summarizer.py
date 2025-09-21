class ExtractiveSummarizer:
    def summarize(self, docs):
        summary = ""
        for doc in docs:
            content = doc["content"].split(".")
            summary += f"- {content}.\n"
        return summary if summary else "No content to summarize."
