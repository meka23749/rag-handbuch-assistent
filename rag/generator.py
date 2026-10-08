"""Generate an answer with an LLM, based only on the retrieved passages.

This is the "G" of RAG: the passages found by the retriever are put into the
prompt, and the LLM is told to answer ONLY from them and to cite the source.
"""
import re
from dataclasses import dataclass, field

import requests

from .retriever import Hit

NOT_FOUND = "Dazu finde ich nichts in den Handbüchern."

SYSTEM_PROMPT = (
    "Du bist ein Assistent für technische Handbücher. "
    "Beantworte die Frage ausschließlich mit Informationen aus den nummerierten Auszügen. "
    "Nenne nach jeder Aussage die Nummer des verwendeten Auszugs in eckigen Klammern, z. B. [1]. "
    f"Wenn die Auszüge die Antwort nicht enthalten, antworte genau: '{NOT_FOUND}' "
    "und nenne dann keine Nummer. "
    "Antworte kurz und auf Deutsch."
)


@dataclass
class Answer:
    text: str                                          # the answer written by the LLM
    passages: list[Hit]                                # all passages given to the LLM
    cited: list[int] = field(default_factory=list)     # numbers [n] the LLM actually cited

    @property
    def found(self) -> bool:
        """False when the LLM said that the manuals do not contain the answer."""
        return not self.text.startswith(NOT_FOUND)

    @property
    def sources(self) -> list[tuple[int, Hit]]:
        """Only the passages that were cited, with their number."""
        if not self.found:
            return []
        return [(n, self.passages[n - 1]) for n in self.cited]


def cited_numbers(text: str, max_number: int) -> list[int]:
    """Find the [n] references in the answer, keep only valid ones, sorted, no duplicates."""
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    return sorted(n for n in numbers if 1 <= n <= max_number)


def build_prompt(question: str, hits: list[Hit]) -> str:
    """Put the passages and the question into one text for the LLM."""
    excerpts = "\n\n".join(
        f"[{i}] {h.source}, Seite {h.page}:\n{h.text}" for i, h in enumerate(hits, start=1)
    )
    return f"Auszüge aus den Handbüchern:\n\n{excerpts}\n\nFrage: {question}"


class OllamaLLM:
    """A local LLM served by Ollama on this computer."""

    def __init__(self, model: str = "qwen2.5:7b", url: str = "http://localhost:11434"):
        self.model = model
        self.url = url

    def complete(self, system: str, user: str) -> str:
        response = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,                  # wait for the whole answer
                "options": {"temperature": 0},    # 0 = no creativity, same answer every time
            },
            timeout=120,
        )
        response.raise_for_status()  # stop with an error if Ollama answered with an error code
        return response.json()["message"]["content"].strip()


def answer(question: str, retriever, llm, k: int = 3) -> Answer:
    """Full RAG: retrieve k passages, then let the LLM answer from them."""
    hits = retriever.search(question, k=k)
    text = llm.complete(SYSTEM_PROMPT, build_prompt(question, hits))
    return Answer(text=text, passages=hits, cited=cited_numbers(text, len(hits)))