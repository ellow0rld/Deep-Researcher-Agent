"""
Research Agent Package
A Python-based research agent that can search, analyze, and synthesize information using local embeddings.
"""

from .embedding import LocalEmbeddingEngine
from .storage import VectorStorage
from .reasoning import MultiStepReasoner
from .summarizer import ExtractiveSummarizer
from .exporter import Exporter
from .agent import ResearchAgent

__all__ = [
    "LocalEmbeddingEngine",
    "VectorStorage",
    "MultiStepReasoner",
    "ExtractiveSummarizer",
    "Exporter",
    "ResearchAgent"
]
