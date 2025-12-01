"""
Benchmark test suites for pyruvector.

Available suites:
- insertion: Vector insertion performance tests
- search: Vector search performance tests
- mixed: Mixed workload performance tests
"""

from .insertion import run as run_insertion
from .search import run as run_search
from .mixed import run as run_mixed

__all__ = [
    'run_insertion',
    'run_search',
    'run_mixed',
]
