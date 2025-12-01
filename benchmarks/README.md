# pyruvector Benchmarks

Comprehensive performance benchmarking suite for pyruvector, measuring vector insertion, search, and advanced operations across different configurations.

## Overview

This benchmark suite includes two types of benchmarks:

### 1. pytest-benchmark Suite (Existing)
- **Vector Operations**: Insert, batch insert, search performance
- **Distance Metrics**: Cosine, Euclidean, DotProduct, Manhattan
- **HNSW Indexing**: Construction and search performance with various parameters
- **Quantization**: Memory and performance impact of Scalar, Product, and Binary quantization
- **Scaling**: Performance characteristics from 1K to 1M+ vectors
- **Metadata Filtering**: Filter query performance with complex conditions
- **Persistence**: Save/load operations with compression

### 2. Database Comparison Suite (NEW)
- **Compare pyruvector vs Qdrant** (and other vector databases)
- **Realistic workloads**: Clustered test vectors mimicking semantic embeddings
- **Comprehensive metrics**: Insertion throughput, search latency, memory, recall
- **Flexible configurations**: Multiple dataset sizes, dimensions, batch sizes
- **Easy to use**: `python3 -m benchmarks.run_benchmarks --quick`

📘 **See [COMPARISON.md](COMPARISON.md) for database comparison documentation**
📘 **See [QUICKSTART.md](QUICKSTART.md) for quick reference**

## Quick Start

### Database Comparison (NEW)

```bash
# Install dependencies
pip install numpy qdrant-client  # qdrant-client optional

# Quick smoke test (30 seconds)
python3 -m benchmarks.run_benchmarks --quick

# Full comparison vs Qdrant
python3 -m benchmarks.run_benchmarks --db all --vectors 10000 50000 100000

# pyruvector only with detailed analysis
python3 -m benchmarks.run_benchmarks \
  --db pyruvector \
  --vectors 10000 50000 100000 \
  --dimensions 128 384 768 \
  --output analysis.json
```

### pytest-benchmark Suite (Existing)

#### Installation

```bash
# Install test dependencies
pip install pyruvector[test]

# Or install manually
pip install pytest pytest-benchmark
```

### Run Benchmarks

```bash
# Run all benchmarks (takes ~5-10 minutes)
pytest benchmarks/ --benchmark-only

# Run quick benchmarks only (1-2 minutes)
pytest benchmarks/ --benchmark-only -m quick

# Run specific benchmark groups
pytest benchmarks/ --benchmark-only -m "insert or search"
pytest benchmarks/ --benchmark-only -m hnsw
pytest benchmarks/ --benchmark-only -m quantization

# Generate detailed HTML report
pytest benchmarks/ --benchmark-only --benchmark-autosave \
    --benchmark-save-data --benchmark-histogram

# Compare against baseline
pytest benchmarks/ --benchmark-only --benchmark-compare
```

### Benchmark Markers

Benchmarks are tagged with pytest markers for selective execution:

- `quick`: Fast benchmarks suitable for CI (<30s total)
- `slow`: Comprehensive benchmarks (>2 minutes)
- `insert`: Vector insertion benchmarks
- `search`: Search operation benchmarks
- `hnsw`: HNSW indexing benchmarks
- `quantization`: Quantization benchmarks
- `persistence`: Save/load benchmarks
- `metadata`: Metadata filtering benchmarks
- `scaling`: Large-scale benchmarks (100K+ vectors)

## Benchmark Organization

```
benchmarks/
├── README.md              # This file
├── conftest.py           # Pytest fixtures and configuration
├── benchmark_insert.py   # Insert and batch operations
├── benchmark_search.py   # Search and retrieval operations
├── benchmark_hnsw.py     # HNSW configuration impact
├── benchmark_quantization.py  # Quantization performance
├── benchmark_persistence.py   # Save/load operations
└── benchmark_scaling.py  # Large-scale performance tests
```

## Understanding Results

### Benchmark Output

```
Name (time in ms)                    Min      Max     Mean    StdDev   Median
--------------------------------------------------------------------------------
test_insert_single[128]            0.015    0.025    0.018    0.003    0.017
test_insert_batch[1000x128]        4.520    5.120    4.780    0.180    4.750
test_search_k10[10000x128]         0.850    1.250    0.920    0.120    0.900
```

**Columns:**
- **Min/Max**: Fastest and slowest run times
- **Mean**: Average time across all runs
- **StdDev**: Standard deviation (lower = more consistent)
- **Median**: Middle value (often more reliable than mean)

### Performance Baselines

Expected performance on modern hardware (M1/M2 MacBook or equivalent):

| Operation | Dataset Size | Expected Time | Notes |
|-----------|--------------|---------------|-------|
| Insert Single | 1 vector | <0.05ms | Per vector overhead |
| Batch Insert | 10K vectors | 50-100ms | HNSW construction |
| Batch Insert | 100K vectors | 600-800ms | HNSW construction |
| Search k=10 | 10K vectors | 0.5-1ms | HNSW O(log n) |
| Search k=10 | 100K vectors | 1-2ms | HNSW O(log n) |
| Search k=10 | 1M vectors | 2-5ms | HNSW O(log n) |
| Save 100K | 100K vectors | 150-250ms | With compression |
| Load 100K | 100K vectors | 100-200ms | Index rebuild |

### Quantization Impact

| Method | Memory | Speed | Recall@10 |
|--------|--------|-------|-----------|
| None (FP32) | 100% | 1.0x | 100% |
| Scalar (8-bit) | 25% | 1.8x | 98% |
| Product (8 subvec) | 12.5% | 2.2x | 95% |
| Binary | 3.1% | 3.5x | 90% |

## Benchmark Options

### pytest-benchmark Arguments

```bash
# Control number of rounds
pytest benchmarks/ --benchmark-min-rounds=10

# Set time limits
pytest benchmarks/ --benchmark-max-time=5.0

# Disable calibration for faster runs
pytest benchmarks/ --benchmark-disable-gc

# Generate comparison
pytest benchmarks/ --benchmark-compare=0001

# Output formats
pytest benchmarks/ --benchmark-json=results.json
pytest benchmarks/ --benchmark-histogram
```

### Environment Variables

```bash
# Set vector dimensions for benchmarks
export BENCHMARK_DIMS=256

# Set dataset sizes
export BENCHMARK_SMALL=1000
export BENCHMARK_MEDIUM=10000
export BENCHMARK_LARGE=100000

# Run quick benchmarks only
export BENCHMARK_QUICK=1
```

## Adding New Benchmarks

### Basic Structure

```python
import pytest
from pyruvector import VectorDB

@pytest.mark.quick
@pytest.mark.search
def test_my_benchmark(benchmark, sample_vectors):
    """Benchmark description."""
    db = VectorDB(dimension=128)
    vectors = sample_vectors(128, 1000)

    # Setup
    for vec in vectors:
        db.insert(vec)

    # Benchmark the operation
    result = benchmark(db.search, vectors[0], k=10)

    # Assertions
    assert len(result) == 10
```

### Using Fixtures

```python
def test_with_populated_db(benchmark, populated_db):
    """Use pre-populated database fixture."""
    query_vector = [0.1] * 128
    result = benchmark(populated_db.search, query_vector, k=10)
    assert len(result) == 10
```

### Parameterized Benchmarks

```python
@pytest.mark.parametrize("dimension", [128, 384, 768, 1536])
@pytest.mark.parametrize("count", [1000, 10000])
def test_scaling(benchmark, dimension, count, sample_vectors):
    """Test scaling across dimensions and counts."""
    db = VectorDB(dimension=dimension)
    vectors = sample_vectors(dimension, count)

    benchmark(db.insert_batch, vectors)
```

## CI Integration

### GitHub Actions Quick Benchmarks

The CI pipeline runs quick benchmarks on every push to main:

```yaml
# .github/workflows/ci.yml
benchmark-quick:
  runs-on: ubuntu-latest
  if: github.ref == 'refs/heads/main'
  steps:
    - name: Run quick benchmarks
      run: pytest benchmarks/ --benchmark-only -m quick
```

### Manual Full Benchmark Workflow

Trigger comprehensive benchmarks manually:

```bash
# Via GitHub Actions UI
# Go to Actions -> Benchmark Suite -> Run workflow

# Or via gh CLI
gh workflow run benchmark.yml
```

### Benchmark Artifacts

Results are uploaded as workflow artifacts:

- **JSON results**: Machine-readable performance data
- **HTML report**: Visual comparison charts
- **Performance baseline**: Reference for regression detection

## Performance Optimization Tips

### HNSW Tuning

```python
from pyruvector import VectorDB, HNSWConfig

# High recall, slower build
config = HNSWConfig(m=48, ef_construction=400, ef_search=200)

# Balanced (default)
config = HNSWConfig(m=16, ef_construction=200, ef_search=50)

# Fast build, lower recall
config = HNSWConfig(m=8, ef_construction=100, ef_search=30)
```

**Guidelines:**
- Increase `m` for better recall (more memory)
- Increase `ef_construction` for better index quality (slower build)
- Increase `ef_search` for better recall (slower search)

### Batch Operations

```python
# ❌ Slow: Individual inserts
for vector in vectors:
    db.insert(vector)

# ✅ Fast: Batch insert (10-100x faster)
db.insert_batch(vectors)
```

### Distance Metric Selection

```python
# Fastest: Pre-normalized vectors
db = VectorDB(dimension=128, distance_metric=DistanceMetric.DotProduct)

# Good: Normalized embeddings (typical use case)
db = VectorDB(dimension=128, distance_metric=DistanceMetric.Cosine)

# Slower: Absolute distances
db = VectorDB(dimension=128, distance_metric=DistanceMetric.Euclidean)
```

### Memory Optimization

```python
# Use quantization for large datasets
from pyruvector import QuantizationConfig, QuantizationType

# 4x memory reduction, minimal accuracy loss
config = QuantizationConfig(
    quantization_type=QuantizationType.Scalar,
    bits=8
)
db = VectorDB(dimension=768, quantization_config=config)
```

## Interpreting Regressions

### Acceptable Variance

- **±5%**: Normal system noise
- **±10%**: Investigate if consistent
- **>20%**: Likely regression, investigate immediately

### Common Causes

1. **System load**: Other processes consuming resources
2. **Thermal throttling**: CPU overheating on long benchmarks
3. **Memory pressure**: Insufficient RAM, swapping
4. **Code changes**: Algorithm modifications
5. **Dependency updates**: Updated libraries with different performance

### Investigation Steps

```bash
# 1. Re-run to confirm
pytest benchmarks/test_specific.py --benchmark-only --benchmark-min-rounds=20

# 2. Compare against baseline
pytest benchmarks/ --benchmark-only --benchmark-compare=baseline

# 3. Profile specific test
python -m cProfile -o profile.stats your_test.py

# 4. Check system resources
top / htop / Activity Monitor
```

## Continuous Performance Monitoring

### Setting Up Baselines

```bash
# Create initial baseline
pytest benchmarks/ --benchmark-only --benchmark-save=baseline

# Compare future runs
pytest benchmarks/ --benchmark-only --benchmark-compare=baseline

# Update baseline
pytest benchmarks/ --benchmark-only --benchmark-save=baseline --benchmark-autosave
```

### Automated Regression Detection

The CI system automatically:

1. Runs quick benchmarks on main branch pushes
2. Uploads results as artifacts
3. (Optional) Posts performance summaries to PRs
4. Fails if performance regresses >25%

## Troubleshooting

### Benchmarks Taking Too Long

```bash
# Use quick benchmarks only
pytest benchmarks/ -m quick --benchmark-only

# Reduce rounds
pytest benchmarks/ --benchmark-min-rounds=3 --benchmark-max-time=1.0

# Skip slow tests
pytest benchmarks/ -m "not slow" --benchmark-only
```

### Inconsistent Results

```bash
# Increase rounds for stability
pytest benchmarks/ --benchmark-min-rounds=20

# Disable GC during benchmarks
pytest benchmarks/ --benchmark-disable-gc

# Close other applications
# Disable background processes
# Check thermal throttling
```

### Out of Memory

```bash
# Run scaling tests individually
pytest benchmarks/benchmark_scaling.py::test_specific --benchmark-only

# Reduce dataset sizes in conftest.py
export BENCHMARK_SMALL=500
export BENCHMARK_MEDIUM=5000
export BENCHMARK_LARGE=50000
```

## Resources

- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [pyruvector Performance Guide](https://github.com/ruvnet/ruvector#performance)

## Contributing

When adding benchmarks:

1. Use appropriate pytest markers (`@pytest.mark.quick`, `@pytest.mark.slow`)
2. Add docstrings explaining what is being measured
3. Use existing fixtures from `conftest.py`
4. Ensure benchmarks are deterministic and reproducible
5. Document expected performance in docstrings
6. Add to this README if introducing new benchmark categories

## License

MIT License - see [LICENSE](../LICENSE) file for details.
