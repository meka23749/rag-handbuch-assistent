"""Read PDF files page by page.

Each page keeps its source (file name) and page number,
so that the assistant can cite where an answer comes from.
"""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    source: str   # file name, e.g. "kompressor_kx200.pdf"
    page: int     # page number, starting at 1 (as a human counts)
    text: str


def load_pdf(path: Path) -> list[Page]:
    reader = PdfReader(str(path))
    pages = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        text = (pdf_page.extract_text() or "").strip()
        if text:  # skip empty pages (e.g. pure images)
            pages.append(Page(source=path.name, page=number, text=text))
    return pages


def load_folder(folder: Path) -> list[Page]:
    pages = []
    for pdf in sorted(folder.glob("*.pdf")):
        pages.extend(load_pdf(pdf))
    return pages