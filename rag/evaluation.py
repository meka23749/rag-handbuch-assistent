"""Measure how well the RAG works on a fixed list of questions.

For every question we know where the answer is (file + page) and which words
a correct answer must contain. Questions without a source are traps: the
assistant must say that the manuals do not contain the answer.

Three measures:
- retrieval hit rate: the right page is among the k passages found
- answer accuracy:    the answer contains all required words and cites the right page
- refusal rate:       trap questions are refused instead of answered
"""
import json
from dataclasses import dataclass
from pathlib import Path

from .generator import answer


@dataclass
class Case:
    question: str
    source: str | None        # None = the answer is NOT in the manuals (trap question)
    page: int | None
    must_contain: list[str]

    @property
    def answerable(self) -> bool:
        return self.source is not None


@dataclass
class Result:
    case: Case
    retrieved: bool           # right page among the passages found
    correct: bool             # answer judged correct (or trap correctly refused)
    answer_text: str


def load_cases(path: Path) -> list[Case]:
    return [Case(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def page_found(hits, case: Case) -> bool:
    """True if the expected file and page are among the hits."""
    return any((h.source, h.page) == (case.source, case.page) for h in hits)


def judge(result_answer, case: Case) -> bool:
    """Decide if an answer is correct for this case."""
    if not case.answerable:
        return not result_answer.found                       # trap: must refuse
    text = result_answer.text.lower()
    has_words = all(word.lower() in text for word in case.must_contain)
    cites_right_page = page_found([hit for _, hit in result_answer.sources], case)
    return result_answer.found and has_words and cites_right_page


def evaluate(cases: list[Case], retriever, llm=None, k: int = 3) -> list[Result]:
    """Run every case. Without an llm, only the retrieval is measured (fast)."""
    results = []
    for case in cases:
        hits = retriever.search(case.question, k=k)
        retrieved = page_found(hits, case) if case.answerable else True
        if llm is None:
            results.append(Result(case, retrieved, correct=False, answer_text=""))
            continue
        result_answer = answer(case.question, retriever, llm, k=k)
        results.append(Result(case, retrieved, judge(result_answer, case), result_answer.text))
    return results


def summary(results: list[Result]) -> dict[str, float]:
    """Percentages over all results."""
    answerable = [r for r in results if r.case.answerable]
    traps = [r for r in results if not r.case.answerable]

    def percent(part, whole):
        return round(100 * len(part) / len(whole), 1) if whole else 0.0

    return {
        "retrieval_hit_rate": percent([r for r in answerable if r.retrieved], answerable),
        "answer_accuracy": percent([r for r in answerable if r.correct], answerable),
        "refusal_rate": percent([r for r in traps if r.correct], traps),
    }