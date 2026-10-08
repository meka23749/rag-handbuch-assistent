"""Turn texts into vectors (embeddings).

An embedding is a list of numbers that represents a text. Texts with a similar
meaning get vectors that point in a similar direction. We measure this with the
cosine similarity: 1 = same direction, 0 = nothing in common.

Two embedders with the same interface (fit / embed / save / load):
- SemanticEmbedder: a pre-trained multilingual model; understands meaning
  ("zu heiß" is close to "zu warm"). Downloads the model once (~0.5 GB).
- TfidfEmbedder: counts words; no download, works offline, but only finds
  texts that share the same words. Good for tests and for comparison.
"""
import math
import pickle
from pathlib import Path

SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class SemanticEmbedder:
    name = "semantic"

    def __init__(self, model_name: str = SEMANTIC_MODEL):
        # imported here, so the rest of the project works without this heavy library
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        """Nothing to learn: the model is already trained."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True -> every vector has length 1
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def save(self, folder: Path) -> None:
        """Nothing to save: the model is loaded again by name."""

    def load(self, folder: Path) -> None:
        """Nothing to load."""


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        # ngram_range=(1, 2): single words and pairs of words ("fehler e42")
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=4096)

    def fit(self, texts: list[str]) -> None:
        """Learn the vocabulary and how rare each word is in our chunks."""
        self.vectorizer.fit(texts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.vectorizer.transform(texts).toarray().tolist()

    def save(self, folder: Path) -> None:
        """Store the learned vocabulary, needed again at search time."""
        with open(folder / "tfidf.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, folder: Path) -> None:
        with open(folder / "tfidf.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)


def make_embedder(kind: str):
    """Create an embedder by name: 'semantic' or 'tfidf'."""
    if kind == "semantic":
        return SemanticEmbedder()
    if kind == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"unknown embedder: {kind}")