from __future__ import annotations

import hashlib
import math


class FaceEmbedder:
    """Small deterministic embedding function for offline-first matching.

    This is intentionally lightweight: callers can swap in a stronger model,
    while tests and Pi-first defaults stay dependency-light.
    """

    def __init__(self, *, dims: int = 32) -> None:
        self._dims = max(8, int(dims))

    def embed(self, face_crop_bytes: bytes) -> list[float]:
        if not face_crop_bytes:
            return []
        vec = [0.0] * self._dims
        digest_chunks = max(1, self._dims // 8)
        cursor = 0
        for idx in range(digest_chunks):
            seed = idx.to_bytes(2, "little")
            digest = hashlib.blake2b(face_crop_bytes + seed, digest_size=32).digest()
            for byte in digest:
                if cursor >= self._dims:
                    break
                vec[cursor] = (byte / 127.5) - 1.0
                cursor += 1
            if cursor >= self._dims:
                break
        return _l2_normalize(vec)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return [0.0 for _ in values]
    return [v / norm for v in values]
