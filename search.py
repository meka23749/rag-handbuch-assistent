"""Ask a question and show the best passages with their source.

Usage:
    python search.py "Was bedeutet Fehler E42?"
    python search.py "Der Laser ist zu heiß" -k 2
"""
import argparse
from pathlib import Path

from rag.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description="Search the manuals.")
    parser.add_argument("question", help="the question, in quotes")
    parser.add_argument("-k", default=3, type=int, help="number of passages to show")
    parser.add_argument("--db", default="data/index", type=Path, help="folder of the database")
    args = parser.parse_args()

    hits = Retriever(args.db).search(args.question, k=args.k)
    for rank, hit in enumerate(hits, start=1):
        print(f"\n[{rank}] {hit.source}, page {hit.page}  (similarity {hit.score})")
        print(f"    {hit.text}")


if __name__ == "__main__":
    main()