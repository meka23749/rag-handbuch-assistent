"""Tests for rag/evaluation.py: judging answers and computing the scores."""
from pathlib import Path

from rag.evaluation import Case, evaluate, judge, load_cases, summary
from rag.generator import NOT_FOUND, Answer
from rag.retriever import Hit

QUESTIONS = Path(__file__).resolve().parent.parent / "eval" / "questions.json"

LASER = Hit(text="L07: Laserquelle zu warm. Anlage abkühlen lassen.", source="laser_lm50.pdf", page=2, score=0.4)
OIL = Hit(text="Öl alle 4000 Betriebsstunden wechseln.", source="kompressor_kx200.pdf", page=2, score=0.3)

LASER_CASE = Case("Laser zu heiß?", "laser_lm50.pdf", 2, ["abkühlen"])
TRAP_CASE = Case("Strompreis?", None, None, [])


def test_question_file_is_valid():
    """The question file loads, and contains both normal and trap questions."""
    cases = load_cases(QUESTIONS)
    assert len(cases) >= 10
    assert any(c.answerable for c in cases) and any(not c.answerable for c in cases)


def test_correct_answer_is_accepted():
    """Required word present and right page cited -> correct."""
    a = Answer(text="Anlage abkühlen lassen [1].", passages=[LASER], cited=[1])
    assert judge(a, LASER_CASE)


def test_missing_word_is_rejected():
    """The right page is cited but the required word is missing -> wrong."""
    a = Answer(text="Lüftungsgitter reinigen [1].", passages=[LASER], cited=[1])
    assert not judge(a, LASER_CASE)


def test_wrong_citation_is_rejected():
    """The words are there, but the cited page is the wrong one -> wrong."""
    a = Answer(text="Anlage abkühlen lassen [2].", passages=[LASER, OIL], cited=[2])
    assert not judge(a, LASER_CASE)


def test_trap_must_be_refused():
    """A trap question is correct only when the assistant refuses."""
    assert judge(Answer(text=NOT_FOUND, passages=[OIL]), TRAP_CASE)
    assert not judge(Answer(text="Der Strom kostet 30 Cent [1].", passages=[OIL], cited=[1]), TRAP_CASE)


class FakeRetriever:
    def search(self, question, k=3):
        return [LASER, OIL][:k]


class FakeLLM:
    def complete(self, system, user):
        return "Anlage abkühlen lassen [1]."


def test_summary_percentages():
    """One normal question answered right and one trap answered wrongly -> 100 % / 0 %."""
    results = evaluate([LASER_CASE, TRAP_CASE], FakeRetriever(), FakeLLM(), k=2)
    assert summary(results) == {"retrieval_hit_rate": 100.0, "answer_accuracy": 100.0, "refusal_rate": 0.0}


def test_retrieval_only_without_llm():
    """Without an LLM only the retrieval is measured."""
    results = evaluate([LASER_CASE], FakeRetriever(), llm=None, k=1)
    assert results[0].retrieved and results[0].answer_text == ""