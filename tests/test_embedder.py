"""Tests for rag/embedder.py: turning texts into vectors."""
import pytest

from rag.embedder import TfidfEmbedder, cosine, make_embedder

TEXTS = [
    "E42: Drehrichtung des Motors falsch.",
    "Der Ansaugfilter ist alle 1000 Betriebsstunden zu prüfen.",
    "L07: Laserquelle zu warm. Anlage abkühlen lassen.",
]


def test_cosine_basics():
    """Same direction gives 1, perpendicular vectors give 0."""
    assert cosine([1, 0], [2, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 3]) == pytest.approx(0.0)


def test_tfidf_vectors_have_same_length():
    """Every text becomes a vector of the same size (needed by the database)."""
    embedder = TfidfEmbedder()
    embedder.fit(TEXTS)
    vectors = embedder.embed(TEXTS)
    assert len({len(v) for v in vectors}) == 1


def test_tfidf_finds_text_with_same_words():
    """A question with the words 'Fehler E42' is closest to the E42 text."""
    embedder = TfidfEmbedder()
    embedder.fit(TEXTS)
    question = embedder.embed(["Was bedeutet E42?"])[0]
    scores = [cosine(question, v) for v in embedder.embed(TEXTS)]
    assert scores.index(max(scores)) == 0


def test_tfidf_save_and_load(tmp_path):
    """After save/load the embedder gives exactly the same vectors."""
    first = TfidfEmbedder()
    first.fit(TEXTS)
    first.save(tmp_path)
    second = TfidfEmbedder()
    second.load(tmp_path)
    assert second.embed(TEXTS) == first.embed(TEXTS)


def test_unknown_embedder_name():
    """An unknown name raises a clear error instead of failing later."""
    with pytest.raises(ValueError):
        make_embedder("unknown")


def test_semantic_understands_synonyms():
    """'zu heiß' is closer to 'zu warm' than to the filter text (needs the model)."""
    pytest.importorskip("sentence_transformers")
    embedder = make_embedder("semantic")
    question, laser, filter_text = embedder.embed(["Der Laser ist zu heiß", TEXTS[2], TEXTS[1]])
    assert cosine(question, laser) > cosine(question, filter_text)