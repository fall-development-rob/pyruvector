"""
PyRuvector Benchmark Framework

A comprehensive benchmarking suite for measuring PyRuvector performance across
various workloads including insertion, search, and mixed operations.

Includes database comparison tools for benchmarking against Qdrant and other vector databases.
"""

from .core.metrics import Timer, MemoryTracker, MetricsCollector
from .core.data_gen import VectorGenerator
from .core.reporter import BenchmarkReporter

__version__ = "1.0.0"
__all__ = [
    "Timer",
    "MemoryTracker",
    "MetricsCollector",
    "VectorGenerator",
    "BenchmarkReporter",
    "compare_vectordbs",
    "run_benchmarks",
]
