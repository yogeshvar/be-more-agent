from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any


@dataclass(slots=True)
class WakeDetectorConfig:
    model_path: str
    threshold: float = 0.5
    trigger_level: int = 2
    refractory_seconds: float = 1.2


class OnnxWakeWordDetector:
    """ONNX wake-word detector tuned for chunked PCM16 input."""

    def __init__(self, config: WakeDetectorConfig) -> None:
        self._config = config
        self._session: Any | None = None
        self._np: Any | None = None
        self._input_name: str | None = None
        self._input_shape: Any | None = None
        self._consecutive_hits = 0
        self._last_trigger_time = 0.0

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        try:
            import numpy as np  # type: ignore[import-not-found]
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Voice wake detector needs numpy and onnxruntime. "
                "Install with: pip install numpy onnxruntime"
            ) from exc
        self._np = np
        self._session = ort.InferenceSession(self._config.model_path, providers=["CPUExecutionProvider"])
        model_inputs = self._session.get_inputs()
        if not model_inputs:
            raise RuntimeError("Wake model has no inputs")
        model_input = model_inputs[0]
        self._input_name = str(model_input.name)
        self._input_shape = getattr(model_input, "shape", None)

    def _prepare_model_input(self, audio_f32: Any) -> Any:
        assert self._np is not None
        rank = len(self._input_shape) if isinstance(self._input_shape, (list, tuple)) else 2
        dims: list[int | None] = []
        if isinstance(self._input_shape, (list, tuple)):
            for dim in self._input_shape:
                if isinstance(dim, int):
                    dims.append(dim)
                else:
                    dims.append(None)
        if rank <= 1:
            return audio_f32
        if rank == 2:
            expected = dims[1] if len(dims) > 1 else None
            if expected and expected > 0:
                flat = audio_f32
                if flat.size < expected:
                    pad = self._np.zeros(expected - flat.size, dtype=self._np.float32)
                    flat = self._np.concatenate([flat, pad])
                elif flat.size > expected:
                    flat = flat[-expected:]
                return self._np.expand_dims(flat, axis=0)
            return self._np.expand_dims(audio_f32, axis=0)
        if rank == 3:
            # Common wake-word export path: [batch, features, frames], e.g. [1, 16, 96].
            feature_dim = dims[1] if len(dims) > 1 else None
            frame_dim = dims[2] if len(dims) > 2 else None
            if feature_dim and frame_dim and feature_dim > 0 and frame_dim > 0:
                needed = feature_dim * frame_dim
                flat = audio_f32
                if flat.size < needed:
                    pad = self._np.zeros(needed - flat.size, dtype=self._np.float32)
                    flat = self._np.concatenate([flat, pad])
                elif flat.size > needed:
                    flat = flat[-needed:]
                return flat.reshape((1, feature_dim, frame_dim))
            return self._np.expand_dims(self._np.expand_dims(audio_f32, axis=0), axis=0)
        shaped = audio_f32
        for _ in range(rank - 1):
            shaped = self._np.expand_dims(shaped, axis=0)
        return shaped

    def _score_chunk(self, pcm16_chunk: bytes) -> float:
        self._ensure_loaded()
        assert self._np is not None
        assert self._session is not None
        assert self._input_name is not None
        if not pcm16_chunk:
            return 0.0
        audio_f32 = self._np.frombuffer(pcm16_chunk, dtype=self._np.int16).astype(self._np.float32) / 32768.0
        if audio_f32.size == 0:
            return 0.0
        model_in = self._prepare_model_input(audio_f32)
        outputs = self._session.run(None, {self._input_name: model_in})
        return self._extract_score(outputs)

    @staticmethod
    def _extract_score(outputs: list[Any]) -> float:
        best = 0.0
        for out in outputs:
            try:
                data = out.ravel()
            except Exception:
                continue
            if getattr(data, "size", 0) == 0:
                continue
            candidate = float(data.max())
            if candidate > best:
                best = candidate
        return best

    def process_chunk(self, pcm16_chunk: bytes) -> bool:
        score = self._score_chunk(pcm16_chunk)
        now = monotonic()
        if score >= self._config.threshold:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0
        if self._consecutive_hits < self._config.trigger_level:
            return False
        if (now - self._last_trigger_time) < self._config.refractory_seconds:
            return False
        self._last_trigger_time = now
        self._consecutive_hits = 0
        return True
