from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GenerationOutput:
    text: str
    input_visual_tokens: int
    output_visual_tokens: int
    prefill_seconds: float  # Deprecated compatibility alias: preparation, not decoder prefill.
    generation_seconds: float
    peak_memory_bytes: int | None
    route_diagnostics: dict[str, Any] = field(default_factory=dict)
    preprocessing_seconds: float = 0.0
    vision_seconds: float = 0.0
    routing_seconds: float = 0.0
    packing_seconds: float = 0.0
    total_seconds: float = 0.0
    selection_overhead_seconds: float = 0.0
    dense_prefill_flops: float | None = None
    routed_prefill_flops: float | None = None
    flops_reduction_percent: float | None = None
    preparation_seconds: float | None = None
    language_generate_seconds: float | None = None
    decoder_prefill_seconds: float | None = None
    decoder_decode_seconds: float | None = None
    timing_protocol: str = 'legacy_unspecified'


class VLMAdapter(ABC):
    @abstractmethod
    def generate(
        self,
        image: str | Path,
        prompt: str,
        method: str,
        budget: int,
        **generation_kwargs: Any,
    ) -> GenerationOutput:
        raise NotImplementedError
