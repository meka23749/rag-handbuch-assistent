"""Tests for rag/chunker.py: splitting pages into chunks."""
from pathlib import Path

from rag.chunker import chunk_page, chunk_pages, split_sentences
from rag.loader import Page, load_folder

PDFS = Path(__file__).resolve().parent.parent / "data" / "pdfs"


def test_split_sentences():
    """A text is cut after . ! and ?, line breaks are removed."""
    text = "Erster Satz.\nZweiter Satz! Dritter?"
    assert split_sentences(text) == ["Erster Satz.", "Zweiter Satz!", "Dritter?"]


def test_error_code_stays_with_its_explanation():
    """A colon does not cut: 'E42: ...' stays in one sentence."""
    sentences = split_sentences("E42: Drehrichtung des Motors falsch. E55: Wartung fällig.")
    assert sentences[0] == "E42: Drehrichtung des Motors falsch."


def test_chunks_are_short_and_keep_the_source():
    """A long page gives several chunks; each keeps file name and page number."""
    page = Page(source="a.pdf", page=3, text=" ".join(f"Das ist Satz {i}." for i in range(50)))
    chunks = chunk_page(page, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.source == "a.pdf" and c.page == 3 for c in chunks)


def test_neighbour_chunks_share_one_sentence():
    """The last sentence of a chunk is repeated at the start of the next one."""
    page = Page(source="a.pdf", page=1, text=" ".join(f"Das ist Satz {i}." for i in range(50)))
    first, second = chunk_page(page, max_chars=200)[:2]
    last_sentence = split_sentences(first.text)[-1]
    assert second.text.startswith(last_sentence)


def test_chunk_ids_are_unique():
    """Every chunk of the sample manuals gets its own id (needed by the database)."""
    chunks = chunk_pages(load_folder(PDFS))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_whole_error_entry_is_in_one_chunk():
    """In the real manual, code E42 and its fix end up in the same chunk."""
    chunks = chunk_pages(load_folder(PDFS))
    e42 = [c for c in chunks if "E42" in c.text]
    assert any("Zwei Phasen" in c.text for c in e42)