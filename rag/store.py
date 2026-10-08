"""Store the chunks and their vectors in a Chroma vector database.

Chroma keeps, for every chunk: its id, its text, its vector and its metadata
(file name and page). The database is saved in a folder on disk, so the
PDFs only need to be processed once.
"""
import json
from pathlib import Path

import chromadb

from .chunker import Chunk
from .embedder import make_embedder

COLLECTION = "handbuecher"  # name of the "table" inside the database


def build_index(chunks: list[Chunk], db_dir: Path, embedder_kind: str) -> int:
    """Embed all chunks and save them in Chroma. Returns the number of chunks."""
    db_dir.mkdir(parents=True, exist_ok=True)

    # 1. turn every chunk text into a vector
    embedder = make_embedder(embedder_kind)
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    embedder.save(db_dir)
    vectors = embedder.embed(texts)

    # 2. open (or create) the database in the folder db_dir
    client = chromadb.PersistentClient(path=str(db_dir))
    # start from scratch each time, so a re-run never creates duplicates
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    # "cosine": compare vectors with the cosine similarity
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    # 3. store everything
    collection.add(
        ids=[c.id for c in chunks],
        documents=texts,
        embeddings=vectors,
        metadatas=[{"source": c.source, "page": c.page} for c in chunks],
    )

    # 4. remember which embedder was used: the questions must use the same one
    info = {"embedder": embedder_kind, "chunks": len(chunks)}
    (db_dir / "index.json").write_text(json.dumps(info))
    return len(chunks)