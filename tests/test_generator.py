"""Tests for rag/generator.py: building the prompt and answering with an LLM."""
import pytest
import requests

from rag.generator import NOT_FOUND, SYSTEM_PROMPT, OllamaLLM, answer, build_prompt, cited_numbers
from rag.retriever import Hit

HITS = [
    Hit(text="L07: Laserquelle zu warm. Anlage abkühlen lassen.", source="laser_lm50.pdf", page=2, score=0.45),
    Hit(text="E42: Drehrichtung des Motors falsch.", source="kompressor_kx200.pdf", page=2, score=0.12),
]


class FakeRetriever:
    """Stands in for the real retriever: always returns the same passages."""

    def search(self, question, k=3):
        return HITS[:k]


class FakeLLM:
    """Stands in for the real LLM: remembers what it received, answers a fixed text."""

    def __init__(self, reply="Anlage abkühlen lassen [1]."):
        self.reply = reply
        self.received = None

    def complete(self, system, user):
        self.received = (system, user)
        return self.reply


def test_prompt_contains_numbered_sources_and_question():
    """Each passage appears with its number, file and page, followed by the question."""
    prompt = build_prompt("Laser zu heiß?", HITS)
    assert "[1] laser_lm50.pdf, Seite 2:" in prompt
    assert "[2] kompressor_kx200.pdf, Seite 2:" in prompt
    assert prompt.endswith("Frage: Laser zu heiß?")


def test_system_prompt_forbids_guessing():
    """The instructions tell the LLM what to say when the answer is not in the passages."""
    assert NOT_FOUND in SYSTEM_PROMPT


def test_answer_passes_passages_to_llm_and_returns_sources():
    """answer() gives the retrieved passages to the LLM and returns them as sources."""
    llm = FakeLLM()
    result = answer("Laser zu heiß?", FakeRetriever(), llm, k=1)
    system, user = llm.received
    assert system == SYSTEM_PROMPT
    assert "L07" in user and "E42" not in user      # only k=1 passage was sent
    assert result.text.startswith("Anlage abkühlen")
    assert [(n, h.source) for n, h in result.sources] == [(1, "laser_lm50.pdf")]

def test_cited_numbers_are_found_and_cleaned():
    """[n] references are extracted once each, sorted; impossible numbers are dropped."""
    assert cited_numbers("A [2]. B [1]. C [2]. D [7].", max_number=3) == [1, 2]


def test_only_cited_passages_are_sources():
    """A passage given to the LLM but not cited is not shown as a source."""
    result = answer("?", FakeRetriever(), FakeLLM("Motor prüfen [2]."), k=2)
    assert [(n, h.source) for n, h in result.sources] == [(2, "kompressor_kx200.pdf")]


def test_no_sources_when_nothing_found():
    """When the LLM says the manuals do not contain the answer, no source is shown."""
    result = answer("Strompreis?", FakeRetriever(), FakeLLM(NOT_FOUND + " [1], [2]"), k=2)
    assert not result.found
    assert result.sources == []


def ollama_running() -> bool:
    """True if the Ollama server answers on this computer."""
    try:
        return requests.get("http://localhost:11434", timeout=1).ok
    except requests.RequestException:
        return False


@pytest.mark.skipif(not ollama_running(), reason="Ollama is not running")
def test_ollama_answers_from_passage():
    """Integration test with the real local model: the answer uses the passage."""
    result = answer("Der Laser ist zu heiß, was soll ich tun?", FakeRetriever(), OllamaLLM(), k=1)
    assert "abkühl" in result.text.lower()