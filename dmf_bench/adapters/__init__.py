"""Benchmark and framework adapter contracts."""

from .base import (
    AnswererAdapter,
    BenchmarkAdapter,
    BenchmarkUnit,
    FrameworkAdapter,
    FrameworkCapability,
    JudgeAdapter,
    LocalFileResource,
    ResumeCapability,
)
from .locomo import LoCoMoAdapter
from .longmemeval import LongMemEvalAdapter

__all__ = [
    "AnswererAdapter",
    "BenchmarkAdapter",
    "BenchmarkUnit",
    "FrameworkAdapter",
    "FrameworkCapability",
    "JudgeAdapter",
    "LocalFileResource",
    "LoCoMoAdapter",
    "LongMemEvalAdapter",
    "ResumeCapability",
]
