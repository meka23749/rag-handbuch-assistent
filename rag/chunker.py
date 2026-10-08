"""Split pages into chunks (small pieces of text).

An LLM answers better with a few short, relevant passages than with a whole
manual. Neighbouring chunks share one sentence (overlap), so that information
cut at a border is not lost.
"""
import re
from dataclasses import dataclass

from .loader import Page


@dataclass
class Chunk:
    id: str       # unique and stable, e.g. "kompressor_kx200.pdf:p2:c0"
    source: str   # file name, copied from the page
    page: int     # page number, copied from the page
    text: str


def split_sentences(text: str) -> list[str]:
    """Cut a text into sentences."""
    # 1. replace line breaks and multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    # 2. cut after . ! or ? when a space and an upper-case letter or a digit follow
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_page(page: Page, max_chars: int = 400, overlap_sentences: int = 1) -> list[Chunk]:
    """Group the sentences of one page into chunks of at most max_chars characters."""
    sentences = split_sentences(page.text)
    texts, current = [], []
    for sentence in sentences:
        # adding this sentence would make the chunk too long -> close the chunk
        if current and len(" ".join(current + [sentence])) > max_chars:
            texts.append(" ".join(current))
            # start the next chunk with the last sentence(s) of the previous one
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(sentence)
    if current:  # the last, unfinished chunk
        texts.append(" ".join(current))
    return [
        Chunk(id=f"{page.source}:p{page.page}:c{i}", source=page.source, page=page.page, text=t)
        for i, t in enumerate(texts)
    ]


def chunk_pages(pages: list[Page], max_chars: int = 400, overlap_sentences: int = 1) -> list[Chunk]:
    """Chunk all pages, one after the other."""
    chunks = []
    for page in pages:
        chunks.extend(chunk_page(page, max_chars, overlap_sentences))
    return chunks