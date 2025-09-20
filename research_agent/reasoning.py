class MultiStepReasoner:
    def __init__(self, summarizer=None):
        self.summarizer = summarizer

    # Keep these if you want for older code
    def break_down_query(self, query):
        import re
        tasks = re.split(r'[.;]', query)
        return [t.strip() for t in tasks if t.strip()]

    def explain_reasoning(self, tasks):
        explanation = "The query is broken down into subtasks:\n"
        for i, t in enumerate(tasks):
            explanation += f"{i+1}. {t}\n"
        return explanation

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
