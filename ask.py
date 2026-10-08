"""Ask a question and get an answer written by the LLM, with sources.

Usage:
    python ask.py "Der Laser ist zu heiß, was soll ich tun?"
    python ask.py "Was bedeutet E42?" --model qwen2.5:3b
"""
import argparse
from pathlib import Path

from rag.generator import OllamaLLM, answer
from rag.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description="Ask the manuals a question.")
    parser.add_argument("question", help="the question, in quotes")
    parser.add_argument("-k", default=3, type=int, help="number of passages given to the LLM")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name")
    parser.add_argument("--db", default="data/index", type=Path, help="folder of the database")
    args = parser.parse_args()

    result = answer(args.question, Retriever(args.db), OllamaLLM(model=args.model), k=args.k)

    print("\nAntwort:\n" + result.text)
    print("\nVerwendete Auszüge:")
    for hit in result.sources:
        print(f"  - {hit.source}, Seite {hit.page} (Ähnlichkeit {hit.score})")
        
if __name__ == "__main__":
    main()