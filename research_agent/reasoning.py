class MultiStepReasoner:
    def __init__(self, summarizer=None):
        self.summarizer = summarizer

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

    def answer_query(self, prompt_text, docs):
        doc_texts = [{"content": d} if isinstance(d, str) else d for d in docs]
        if self.summarizer:
            return self.summarizer.summarize(doc_texts)
        else:
            return "\n".join([d["content"].split(".")[0] + "." for d in doc_texts])
