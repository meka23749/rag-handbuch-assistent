"""Tests for rag/retriever.py: finding the right passage for a question."""
from pathlib import Path

import pytest

from rag.chunker import chunk_pages
from rag.loader import load_folder
from rag.retriever import Retriever
from rag.store import build_index

PDFS = Path(__file__).resolve().parent.parent / "data" / "pdfs"


def make_retriever(db_dir, embedder_kind):
    """Helper: build an index of the sample manuals and open a retriever on it."""
    build_index(chunk_pages(load_folder(PDFS)), db_dir, embedder_kind)
    return Retriever(db_dir)


@pytest.fixture(scope="module")
def tfidf_retriever(tmp_path_factory):
    """One tf-idf index shared by the tests of this file (built only once)."""
    return make_retriever(tmp_path_factory.mktemp("tfidf"), "tfidf")


def test_returns_k_hits_best_first(tfidf_retriever):
    """We get exactly k results, sorted from most to least similar."""
    hits = tfidf_retriever.search("Wartung Filter", k=3)
    assert len(hits) == 3
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)


@pytest.mark.parametrize("question, source, page", [
    ("Was bedeutet Fehler E42?", "kompressor_kx200.pdf", 2),
    ("Wann wird der Ansaugfilter gewechselt?", "kompressor_kx200.pdf", 2),
    ("Was tun bei Meldung L07?", "laser_lm50.pdf", 2),
])
def test_tfidf_finds_right_page_when_words_match(tfidf_retriever, question, source, page):
    """With the same words as in the manual, tf-idf finds the right page."""
    best = tfidf_retriever.search(question, k=1)[0]
    assert (best.source, best.page) == (source, page)


def test_semantic_finds_synonym(tmp_path):
    """'zu heiß' finds 'Laserquelle zu warm' although the word 'heiß' never appears."""
    pytest.importorskip("sentence_transformers")
    retriever = make_retriever(tmp_path, "semantic")
    best = retriever.search("Der Laser ist zu heiß, was soll ich tun?", k=1)[0]
    assert best.source == "laser_lm50.pdf" and "L07" in best.text