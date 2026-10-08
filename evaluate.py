"""Evaluate the RAG on eval/questions.json and print a report.

Usage:
    python evaluate.py --retrieval-only        # fast, no LLM
    python evaluate.py                         # full: retrieval + answers (uses Ollama)
    python evaluate.py -k 1 --retrieval-only   # compare settings
"""
import argparse
from pathlib import Path

from rag.evaluation import evaluate, load_cases, summary
from rag.generator import OllamaLLM
from rag.retriever import Retriever


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG.")
    parser.add_argument("--questions", default="eval/questions.json", type=Path)
    parser.add_argument("--db", default="data/index", type=Path)
    parser.add_argument("-k", default=3, type=int, help="number of passages")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM")
    args = parser.parse_args()

    llm = None if args.retrieval_only else OllamaLLM(model=args.model)
    results = evaluate(load_cases(args.questions), Retriever(args.db), llm, k=args.k)

    for r in results:
        kind = "Frage" if r.case.answerable else "Falle"
        ok_retrieval = "ok" if r.retrieved else "NEIN"
        ok_answer = "-" if llm is None else ("ok" if r.correct else "FALSCH")
        print(f"[{kind}] Suche: {ok_retrieval:4}  Antwort: {ok_answer:6}  {r.case.question}")
        if llm is not None and not r.correct:
            print(f"         -> {r.answer_text}")

    print("\nErgebnis:")
    for name, value in summary(results).items():
        if llm is None and name != "retrieval_hit_rate":
            continue
        print(f"  {name}: {value} %")


if __name__ == "__main__":
    main()