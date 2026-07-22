"""BGE-M3 embeddings with contextual prefixes for master index assembly."""

from __future__ import annotations

import gc
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from crosscheck.config import EMBEDDING_MODEL, get_embedding_device_pref
from crosscheck.models import Chunk


def chunk_embedding_text(chunk: Chunk) -> str:
    """Build the contextual embedding string for BGE-M3."""
    period = f"{chunk.fiscal_period}{chunk.fiscal_year}"
    if chunk.doc_type == "transcript":
        # Speaker/role matter for retrieval context; prepared vs Q&A section does not.
        speaker = chunk.speaker_name or "Unknown"
        role = chunk.speaker_role or "Unknown"
        header = (
            f"[COMPANY: {chunk.ticker} | PERIOD: {period} | TYPE: {chunk.doc_type} | "
            f"SPEAKER: {speaker} ({role})]"
        )
    else:
        section = chunk.section or "Unknown"
        header = (
            f"[COMPANY: {chunk.ticker} | PERIOD: {period} | TYPE: {chunk.doc_type} | "
            f"SECTION: {section} | TABLE: {chunk.is_table}]"
        )
    return f"{header}\n{chunk.text.strip()}"


def resolve_device() -> str:
    """Pick ``mps``, ``cuda``, or ``cpu`` based on env and availability."""
    pref = get_embedding_device_pref()
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and torch.backends.mps.is_available():
        return "mps"
    if pref == "cuda" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    """Load and cache the sentence-transformer model."""
    name = model_name or EMBEDDING_MODEL
    device = resolve_device()
    print(f"[embeddings] loading {name} on {device}", flush=True)
    return SentenceTransformer(name, device=device)


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int = 16,
) -> np.ndarray:
    """Encode normalized vectors in small batches and purge accelerator memory.

    Every batch is moved to a CPU NumPy array immediately. This avoids retaining
    MPS tensors across the full corpus and mitigates PyTorch MPS memory growth.
    """
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    cpu_batches: list[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[start : start + batch_size]
        encoded: torch.Tensor | None = None
        try:
            with torch.no_grad():
                encoded = model.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=False,
                    convert_to_tensor=True,
                )

            # Detach from MPS/CUDA immediately and retain only CPU float32 data.
            cpu_array = (
                encoded.detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
                .copy()
            )
            cpu_batches.append(cpu_array)
        finally:
            del encoded
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        if total_batches > 1 and (
            batch_number == 1
            or batch_number % 10 == 0
            or batch_number == total_batches
        ):
            done = min(start + len(batch), len(texts))
            print(
                f"  embedded batch {batch_number}/{total_batches} "
                f"({done}/{len(texts)} chunks)",
                flush=True,
            )

    # IndexFlatIP expects a contiguous float32 matrix. Vectors are already L2
    # normalized, so inner product is cosine similarity.
    return np.ascontiguousarray(np.vstack(cpu_batches), dtype=np.float32)
