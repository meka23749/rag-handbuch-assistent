"""Read all PDFs of a folder and build the search index.

Usage:
    python ingest.py                       # data/pdfs, semantic embeddings
    python ingest.py --embedder tfidf      # offline, no model download
"""
import argparse
from pathlib import Path

from rag.chunker import chunk_pages
from rag.loader import load_folder
from rag.store import build_index


def main():
    parser = argparse.ArgumentParser(description="Build the search index from PDFs.")
    parser.add_argument("--pdfs", default="data/pdfs", type=Path, help="folder with the PDFs")
    parser.add_argument("--db", default="data/index", type=Path, help="folder for the database")
    parser.add_argument("--embedder", default="semantic", choices=["semantic", "tfidf"])
    parser.add_argument("--max-chars", default=400, type=int, help="maximum chunk size")
    args = parser.parse_args()

    pages = load_folder(args.pdfs)
    if not pages:
        raise SystemExit(f"No PDF text found in {args.pdfs}")
    chunks = chunk_pages(pages, max_chars=args.max_chars)
    count = build_index(chunks, args.db, args.embedder)

    files = len({p.source for p in pages})
    print(f"{files} PDF(s), {len(pages)} pages, {count} chunks -> {args.db} ({args.embedder})")


if __name__ == "__main__":
    main()