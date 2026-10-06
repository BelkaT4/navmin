"""Reference memory for VT-10 HOTFIX-5.

The immutable INITIAL TRAIN descriptors form ``BaseBank``.  Adaptive memory is
kept in separate layers:

* ``adaptive_descriptor`` -- slow EMA (kept for VT-9 compatibility);
* ``fast_descriptor`` -- fast drift-response EMA;
* ``recent_descriptors`` -- a small FIFO of recent trusted appearances.

No adaptive layer may evict or modify BaseBank entries.  Matching is best-of-
memory, while long-term drift estimates can still be computed against the base
and slow layers only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Iterable, Mapping

from navmin.vision.processors.vt11.identity.descriptor import DescriptorSimilarity, TargetDescriptor, descriptor_similarity


@dataclass
class ReferenceBank:
    max_size: int = 5
    descriptors: list[TargetDescriptor] = field(default_factory=list)
    adaptive_descriptor: TargetDescriptor | None = None  # slow EMA
    fast_descriptor: TargetDescriptor | None = None
    recent_descriptors: list[TargetDescriptor] = field(default_factory=list)
    recent_max_size: int = 5

    def __post_init__(self) -> None:
        if self.max_size < 1:
            raise ValueError("max_size must be >= 1")
        if self.recent_max_size < 1:
            raise ValueError("recent_max_size must be >= 1")
        if len(self.recent_descriptors) > self.recent_max_size:
            self.recent_descriptors = self.recent_descriptors[-self.recent_max_size:]

    def __len__(self) -> int:
        return (
            len(self.descriptors)
            + (1 if self.adaptive_descriptor is not None else 0)
            + (1 if self.fast_descriptor is not None else 0)
            + len(self.recent_descriptors)
        )

    @property
    def base_count(self) -> int:
        return len(self.descriptors)

    @property
    def effective_count(self) -> int:
        return len(self)

    @property
    def recent_count(self) -> int:
        return len(self.recent_descriptors)

    def clear(self) -> None:
        self.descriptors.clear()
        self.clear_adaptive()

    def clear_adaptive(self) -> None:
        self.adaptive_descriptor = None
        self.fast_descriptor = None
        self.recent_descriptors.clear()

    def set_adaptive(self, descriptor: TargetDescriptor) -> None:
        """Set slow EMA layer (legacy VT-9 API)."""
        self.adaptive_descriptor = descriptor

    def set_fast(self, descriptor: TargetDescriptor | None) -> None:
        self.fast_descriptor = descriptor

    def add_recent(self, descriptor: TargetDescriptor, *, max_size: int | None = None) -> None:
        if max_size is not None:
            if max_size < 1:
                raise ValueError("recent max_size must be >= 1")
            self.recent_max_size = int(max_size)
        self.recent_descriptors.append(descriptor)
        if len(self.recent_descriptors) > self.recent_max_size:
            del self.recent_descriptors[:-self.recent_max_size]

    def add(self, descriptor: TargetDescriptor) -> None:
        self.descriptors.append(descriptor)
        if len(self.descriptors) > self.max_size:
            self.descriptors.pop(0)

    def extend(self, descriptors: Iterable[TargetDescriptor]) -> None:
        for descriptor in descriptors:
            self.add(descriptor)

    def clone(self, *, include_adaptive: bool = True) -> "ReferenceBank":
        return ReferenceBank(
            max_size=self.max_size,
            descriptors=list(self.descriptors),
            adaptive_descriptor=self.adaptive_descriptor if include_adaptive else None,
            fast_descriptor=self.fast_descriptor if include_adaptive else None,
            recent_descriptors=list(self.recent_descriptors) if include_adaptive else [],
            recent_max_size=self.recent_max_size,
        )

    def _effective_descriptors(self) -> list[TargetDescriptor]:
        refs = list(self.descriptors)
        if self.adaptive_descriptor is not None:
            refs.append(self.adaptive_descriptor)
        if self.fast_descriptor is not None:
            refs.append(self.fast_descriptor)
        refs.extend(self.recent_descriptors)
        return refs

    def _long_term_descriptors(self) -> list[TargetDescriptor]:
        refs = list(self.descriptors)
        if self.adaptive_descriptor is not None:
            refs.append(self.adaptive_descriptor)
        return refs

    def expected_size_px(self) -> float:
        """Current target-area estimate.

        Recent trusted states are preferred because scale can change quickly.
        A median of recent samples is more robust than using one sample.  If no
        recent states exist, fast EMA, slow EMA, then BaseBank are used.
        """
        if self.recent_descriptors:
            return float(median(max(1.0, d.size_px) for d in self.recent_descriptors))
        if self.fast_descriptor is not None:
            return float(max(1.0, self.fast_descriptor.size_px))
        if self.adaptive_descriptor is not None:
            return float(max(1.0, self.adaptive_descriptor.size_px))
        if not self.descriptors:
            raise ValueError("reference bank is empty")
        return float(median(max(1.0, d.size_px) for d in self.descriptors))

    def best_base_match(
        self,
        descriptor: TargetDescriptor,
        *,
        identity_weights: Mapping[str, float] | None = None,
    ) -> tuple[TargetDescriptor, DescriptorSimilarity]:
        if not self.descriptors:
            raise ValueError("reference bank is empty")
        pairs = [
            (ref, descriptor_similarity(ref, descriptor, identity_weights=identity_weights))
            for ref in self.descriptors
        ]
        return max(pairs, key=lambda pair: pair[1].total)

    def best_long_term_match(
        self,
        descriptor: TargetDescriptor,
        *,
        identity_weights: Mapping[str, float] | None = None,
    ) -> tuple[TargetDescriptor, DescriptorSimilarity]:
        refs = self._long_term_descriptors()
        if not refs:
            raise ValueError("reference bank is empty")
        pairs = [
            (ref, descriptor_similarity(ref, descriptor, identity_weights=identity_weights))
            for ref in refs
        ]
        return max(pairs, key=lambda pair: pair[1].total)

    def best_match(
        self,
        descriptor: TargetDescriptor,
        *,
        identity_weights: Mapping[str, float] | None = None,
    ) -> tuple[TargetDescriptor, DescriptorSimilarity]:
        refs = self._effective_descriptors()
        if not refs:
            raise ValueError("reference bank is empty")
        pairs = [
            (ref, descriptor_similarity(ref, descriptor, identity_weights=identity_weights))
            for ref in refs
        ]
        return max(pairs, key=lambda pair: pair[1].total)

    def similarity(
        self,
        descriptor: TargetDescriptor,
        *,
        identity_weights: Mapping[str, float] | None = None,
    ) -> DescriptorSimilarity:
        _ref, similarity = self.best_match(descriptor, identity_weights=identity_weights)
        return similarity
