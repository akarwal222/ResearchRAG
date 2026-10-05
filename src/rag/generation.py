"""Generation: prompt construction and local LLM inference through Ollama."""

import logging

import httpx
import ollama

from rag.config import Settings, get_settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research assistant that answers questions about machine learning papers.

Rules:
1. Use ONLY the numbered context passages provided. Do not use outside knowledge.
2. Cite the passages that support each claim by number, like [1] or [2][3].
3. If the context does not contain the answer, reply exactly: "I could not find the answer in the provided papers."
4. Be concise and precise. Keep technical terms and notation as written in the context."""


def build_messages(question: str, context: str) -> list[dict]:
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class GenerationError(RuntimeError):
    """Raised when the LLM backend is unreachable or returns an error."""


def _model_names(resp) -> list[str]:
    """Extract model names from client.list(), across ollama-python versions."""
    models = getattr(resp, "models", None)
    if models is None:
        models = resp.get("models", [])
    names = []
    for m in models:
        if isinstance(m, dict):
            name = m.get("model") or m.get("name")
        else:
            name = getattr(m, "model", None)
        if name:
            names.append(name)
    return names


class OllamaGenerator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.client = ollama.Client(
            host=self.settings.ollama_host, timeout=self.settings.llm_timeout_s
        )

    def generate(self, messages: list[dict]) -> str:
        try:
            resp = self.client.chat(
                model=self.settings.llm_model,
                messages=messages,
                keep_alive=self.settings.llm_keep_alive,
                options={
                    "temperature": self.settings.llm_temperature,
                    "num_ctx": self.settings.llm_num_ctx,
                },
            )
        except (ollama.ResponseError, httpx.HTTPError, ConnectionError) as exc:
            raise GenerationError(
                f"Ollama request failed ({self.settings.ollama_host}): {exc}"
            ) from exc
        return resp["message"]["content"].strip()

    def warmup(self) -> None:
        """Load the model into VRAM now rather than on the first question."""
        try:
            self.client.generate(
                model=self.settings.llm_model,
                prompt="",
                keep_alive=self.settings.llm_keep_alive,
            )
        except Exception as exc:
            log.warning("LLM warmup failed: %s", exc)

    def status(self) -> tuple[bool, bool]:
        """Returns (ollama_reachable, configured_model_is_pulled)."""
        try:
            names = _model_names(self.client.list())
        except Exception:
            return False, False
        model = self.settings.llm_model
        return True, model in names or f"{model}:latest" in names
