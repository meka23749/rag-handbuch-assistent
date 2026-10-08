"""Tests for rag/loader.py: reading PDFs page by page."""
from pathlib import Path

from rag.loader import load_folder, load_pdf

# Folder with the sample manuals, found relative to this test file
# (tests/ -> project root -> data/pdfs), so the tests work from any directory.
PDFS = Path(__file__).resolve().parent.parent / "data" / "pdfs"


def test_load_pdf_keeps_source_and_page_numbers():
    """Each page keeps its file name and its page number, starting at 1."""
    pages = load_pdf(PDFS / "kompressor_kx200.pdf")
    # the compressor manual has 2 pages, numbered like a human counts
    assert [p.page for p in pages] == [1, 2]
    # every page remembers which file it comes from
    assert all(p.source == "kompressor_kx200.pdf" for p in pages)


def test_error_code_is_on_page_two():
    """The text lands on the correct page (needed later to cite the source)."""
    pages = load_pdf(PDFS / "kompressor_kx200.pdf")
    assert "E42" in pages[1].text       # pages[1] = second page (Python counts from 0)
    assert "E42" not in pages[0].text   # and not on the first page


def test_load_folder_reads_all_pdfs():
    """All PDFs of the folder are read, none is forgotten."""
    sources = {p.source for p in load_folder(PDFS)}  # set = each file name once
    assert sources == {"kompressor_kx200.pdf", "laser_lm50.pdf"}