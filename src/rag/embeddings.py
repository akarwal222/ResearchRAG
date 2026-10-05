"""Thin wrapper around sentence-transformers.

The model loads lazily on first use, so importing this module is cheap and
unit tests that never embed do not pay the load cost.
"""

import numpy as np

from rag.config import Settings, get_settings


class Embedder:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = self.settings.embedding_device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(self.settings.embedding_model, device=device)
            if device == "cuda":
                model.half()  # fp16: about 2x faster, negligible quality change
            self._model = model
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Documents are embedded as-is, without the BGE query instruction."""
        return self.model.encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            self.settings.query_instruction + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
