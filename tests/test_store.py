"""Tests for rag/store.py: building the Chroma index."""
import json
from pathlib import Path

import chromadb

from rag.chunker import chunk_pages
from rag.loader import load_folder
from rag.store import COLLECTION, build_index

PDFS = Path(__file__).resolve().parent.parent / "data" / "pdfs"


def open_collection(db_dir):
    """Helper: open the collection that build_index wrote."""
    return chromadb.PersistentClient(path=str(db_dir)).get_collection(COLLECTION)


def test_every_chunk_is_stored(tmp_path):
    """The database contains exactly as many entries as there are chunks."""
    chunks = chunk_pages(load_folder(PDFS))
    count = build_index(chunks, tmp_path, "tfidf")
    assert count == len(chunks)
    assert open_collection(tmp_path).count() == len(chunks)


def test_metadata_is_kept(tmp_path):
    """File name and page number are stored next to each chunk."""
    chunks = chunk_pages(load_folder(PDFS))
    build_index(chunks, tmp_path, "tfidf")
    stored = open_collection(tmp_path).get(ids=[chunks[0].id])
    assert stored["metadatas"][0] == {"source": chunks[0].source, "page": chunks[0].page}


def test_rebuild_does_not_duplicate(tmp_path):
    """Running the indexing twice gives the same number of entries, not double."""
    chunks = chunk_pages(load_folder(PDFS))
    build_index(chunks, tmp_path, "tfidf")
    build_index(chunks, tmp_path, "tfidf")
    assert open_collection(tmp_path).count() == len(chunks)


def test_embedder_is_remembered(tmp_path):
    """index.json records which embedder was used."""
    build_index(chunk_pages(load_folder(PDFS)), tmp_path, "tfidf")
    info = json.loads((tmp_path / "index.json").read_text())
    assert info["embedder"] == "tfidf"