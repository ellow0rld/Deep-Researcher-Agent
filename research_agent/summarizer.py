from collections import Counter
import re

class ExtractiveSummarizer:
    """
    Summarizes multiple documents by selecting key sentences.
    """
    def summarize(self, documents, max_sentences=5):
        all_text = " ".join([doc["content"] for doc in documents])
        sentences = re.split(r'(?<=[.!?]) +', all_text)
        # simple scoring: sentence length + word frequency
        words = [w.lower() for s in sentences for w in s.split()]
        freqs = Counter(words)
        scored = []
        for s in sentences:
            score = sum(freqs.get(w.lower(),0) for w in s.split())
            scored.append((score, s))
        scored.sort(reverse=True)
        top_sentences = [s for _, s in scored[:max_sentences]]
        return " ".join(top_sentences)
