"""Find the chunks that best match a question.

The question is turned into a vector with the SAME embedder as the chunks,
then Chroma returns the k closest chunks with their source and page.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import chromadb

from .embedder import make_embedder
from .store import COLLECTION


@dataclass
class Hit:
    text: str
    source: str
    page: int
    score: float  # cosine similarity: closer to 1 = more similar


class Retriever:
    def __init__(self, db_dir: Path):
        # read which embedder built the index, and load the same one
        info = json.loads((db_dir / "index.json").read_text())
        self.embedder = make_embedder(info["embedder"])
        self.embedder.load(db_dir)
        self.collection = chromadb.PersistentClient(path=str(db_dir)).get_collection(COLLECTION)

    def search(self, question: str, k: int = 3) -> list[Hit]:
        """Return the k chunks closest to the question, best first."""
        result = self.collection.query(
            query_embeddings=self.embedder.embed([question]),
            n_results=k,
        )
        # Chroma answers for a list of questions; we asked one, so we take [0]
        texts, metas, distances = result["documents"][0], result["metadatas"][0], result["distances"][0]
        return [
            # Chroma gives a cosine DISTANCE (0 = identical); similarity = 1 - distance
            Hit(text=t, source=m["source"], page=m["page"], score=round(1 - d, 3))
            for t, m, d in zip(texts, metas, distances)
        ]