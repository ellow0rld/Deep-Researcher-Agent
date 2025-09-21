class MultiStepReasoner:
    def __init__(self, summarizer=None):
        self.summarizer = summarizer

    # Keep these if you want for older code
    def break_down_query(self, query):
        import re
        q = query.lower().strip()
        parts = re.split(r"\band\b|\bor\b|,|\?|;", q)
        parts = [p.strip().capitalize() for p in parts if p.strip()]
        if len(parts) <= 1:
            return [query]
        return parts

    def explain_reasoning(self, tasks):
        reasoning = []
        for i, task in enumerate(tasks, 1):
            reasoning.append(f"{i}. {task}")
        return "\n".join(reasoning)

    # NEW: generate a meaningful answer using retrieved docs
    def answer_query(self, prompt_text, docs):
        """
        Produce a single coherent answer given prompt_text and a list of docs.
        Currently uses ExtractiveSummarizer; can be replaced with an LLM.
        """
        doc_texts = [{"content": d} if isinstance(d, str) else d for d in docs]
        if self.summarizer:
            return self.summarizer.summarize(doc_texts)
        else:
            # fallback: concatenate first sentences
            return "\n".join([d["content"].split(".")[0] + "." for d in doc_texts])

