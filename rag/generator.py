"""Generate an answer with an LLM, based only on the retrieved passages.

This is the "G" of RAG: the passages found by the retriever are put into the
prompt, and the LLM is told to answer ONLY from them and to cite the source.
"""
from dataclasses import dataclass

import requests

from .retriever import Hit

SYSTEM_PROMPT = (
    "Du bist ein Assistent für technische Handbücher. "
    "Beantworte die Frage ausschließlich mit Informationen aus den nummerierten Auszügen. "
    "Nenne nach jeder Aussage die Quelle im Format [Datei, Seite X]. "
    "Wenn die Auszüge die Antwort nicht enthalten, antworte genau: "
    "'Dazu finde ich nichts in den Handbüchern.' "
    "Antworte kurz und auf Deutsch."
)


@dataclass
class Answer:
    text: str          # the answer written by the LLM
    sources: list[Hit]  # the passages that were given to the LLM


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
    return Answer(text=text, sources=hits)