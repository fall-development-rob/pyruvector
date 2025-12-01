# pyruvector
**by Rob-otix Ai Ltd**

[![PyPI version](https://badge.fury.io/py/pyruvector.svg)](https://badge.fury.io/py/pyruvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Crates.io](https://img.shields.io/crates/v/ruvector-core.svg)](https://crates.io/crates/ruvector-core)
[![Rob-otix AI](https://img.shields.io/badge/Rob--otix-AI%20Ltd-blueviolet.svg)](https://rob-otix.ai)

**A distributed vector database that learns.** Store embeddings, query with Cypher, scale horizontally with Raft consensus, and let the index improve itself through Graph Neural Networks.

Python bindings for [rUvector](https://github.com/ruvnet/ruvector) — the Rust vector database ecosystem, maintained and enhanced by Rob-otix Ai Ltd.

```bash
pip install pyruvector
```

**Think of it as: Pinecone + Neo4j + PyTorch + etcd in one Python package.**

## What Problem Does pyruvector Solve?

Traditional vector databases just store and search. When you ask "find similar items," they return results but never get smarter. They don't scale horizontally. They can't route AI requests intelligently.

**pyruvector is different:**

- **Store vectors** like any vector DB (embeddings from OpenAI, Cohere, etc.)
- **Query with Cypher** like Neo4j (`MATCH (a)-[:SIMILAR]->(b) RETURN b`)
- **The index learns** — GNN layers make search results improve over time
- **Scale horizontally** — Raft consensus, multi-master replication, auto-sharding
- **Route AI requests** — Semantic routing and FastGRNN neural inference for LLM optimization
- **Compress automatically** — 2-32x memory reduction with adaptive tiered compression

## Features

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **Vector Search** | HNSW index, <1ms latency, SIMD acceleration | Fast enough for real-time apps |
| **Cypher Queries** | MATCH, WHERE, CREATE, RETURN | Familiar Neo4j syntax |
| **GNN Layers** | Neural network on index topology | Search improves with usage |
| **Hyperedges** | Connect 3+ nodes at once | Model complex relationships |
| **Metadata Filtering** | Filter vectors by properties | Combine semantic + structured search |
| **Collections** | Namespace isolation, multi-tenancy | Organize vectors by project/user |

### Distributed Systems

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **Raft Consensus** | Leader election, log replication | Strong consistency for metadata |
| **Auto-Sharding** | Consistent hashing, shard migration | Scale to billions of vectors |
| **Multi-Master Replication** | Write to any node, conflict resolution | High availability, no SPOF |
| **Snapshots** | Point-in-time backups, incremental | Disaster recovery |
| **Cluster Metrics** | Prometheus-compatible monitoring | Observability at scale |

### AI & ML

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **Tensor Compression** | Scalar/Product/Binary quantization | 2-32x memory reduction |
| **Semantic Router** | Route queries to optimal endpoints | Multi-model AI orchestration |
| **Tiny Dancer** | FastGRNN neural inference | Optimize LLM inference costs |
| **Adaptive Routing** | Learn optimal routing strategies | Minimize latency, maximize accuracy |

## Comparison with Alternatives

| Feature | pyruvector | Pinecone | Weaviate | Qdrant |
|---------|-----------|----------|----------|--------|
| HNSW Indexing | ✅ | ✅ | ✅ | ✅ |
| Cypher Queries | ✅ | ❌ | ❌ | ❌ |
| GNN-Enhanced Search | ✅ | ❌ | ❌ | ❌ |
| Hyperedges | ✅ | ❌ | ❌ | ❌ |
| Raft Consensus | ✅ | ⚡ managed | ✅ | ✅ |
| Multi-Master | ✅ | ❌ | ✅ | ❌ |
| AI Routing | ✅ | ❌ | ❌ | ❌ |
| Quantization | ✅ | ✅ | ✅ | ✅ |
| Self-Hosted | ✅ | ❌ | ✅ | ✅ |
| Open Source | ✅ | ❌ | ✅ | ✅ |

## Use Cases

### RAG (Retrieval-Augmented Generation)
```python
from pyruvector import VectorDB, DistanceMetric

# Store document embeddings
rag_db = VectorDB(dimension=1536, distance_metric=DistanceMetric.Cosine)
rag_db.insert(doc_embedding, {"text": "...", "source": "document.pdf"})

# Retrieve relevant context for LLM
results = rag_db.search(query_embedding, k=5)
context = "\n".join([r["metadata"]["text"] for r in results])
```

### Recommendation Systems
```python
# Product recommendations with metadata filtering
results = db.search(
    user_preference_vector,
    k=20,
    filter={"category": {"$in": ["electronics", "gadgets"]}, "price": {"$lt": 500}}
)
```

### Knowledge Graphs
```python
from pyruvector import GraphDB

# Create knowledge graph with Cypher
graph = GraphDB()
alice = graph.create_node("Person", {"name": "Alice", "age": 30})
bob = graph.create_node("Person", {"name": "Bob", "age": 28})
graph.create_edge(alice, bob, "KNOWS", {"since": "2020"})

# Query with Cypher
results = graph.query("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b")
```

### Semantic Search
```python
# Search with automatic embedding-based similarity
results = db.search(
    query_embedding,
    k=10,
    filter={"language": {"$eq": "en"}, "published": {"$exists": True}}
)
```

## Installation

### From PyPI (Recommended)

```bash
pip install pyruvector
```

### From Source

```bash
# Install maturin
pip install maturin

# Clone the repository
git clone https://github.com/ruvnet/ruvector.git
cd ruvector/pyruvector

# Build and install
maturin develop --release
```

### Requirements

- Python 3.9 or higher
- pip 21.0 or higher (for binary wheel support)

## Quick Start

```python
from pyruvector import VectorDB, DistanceMetric

# Create a vector database with HNSW indexing and Euclidean distance
db = VectorDB(dimension=4, distance_metric=DistanceMetric.Euclidean)

# Insert vectors with metadata
db.insert([1.0, 0.0, 0.0, 0.0], {"category": "A", "value": 10})
db.insert([0.0, 1.0, 0.0, 0.0], {"category": "B", "value": 20})
db.insert([0.0, 0.0, 1.0, 0.0], {"category": "A", "value": 30})

# Search for similar vectors
results = db.search([0.9, 0.1, 0.0, 0.0], k=2)

for result in results:
    print(f"ID: {result['id']}, Distance: {result['distance']:.4f}")
    print(f"Metadata: {result['metadata']}")

# Save to disk
db.save("my_vectors.db")

# Load from disk
db = VectorDB.load("my_vectors.db")
```

## Configuration

### Distance Metrics

Choose the appropriate distance metric for your use case:

```python
from pyruvector import VectorDB, DistanceMetric

# Cosine similarity (default) - normalized dot product, range [0, 2]
db = VectorDB(dimension=128, distance_metric=DistanceMetric.Cosine)

# Euclidean distance - L2 norm, measures absolute distance
db = VectorDB(dimension=128, distance_metric=DistanceMetric.Euclidean)

# Dot product - raw inner product, higher is more similar
db = VectorDB(dimension=128, distance_metric=DistanceMetric.DotProduct)

# Manhattan distance - L1 norm, city block distance
db = VectorDB(dimension=128, distance_metric=DistanceMetric.Manhattan)
```

**Distance Metric Guide:**
- **Cosine**: Best for normalized embeddings (e.g., text embeddings from OpenAI, Cohere)
- **Euclidean**: Best for absolute distance measurements (e.g., spatial data, image features)
- **DotProduct**: Best for unnormalized embeddings where magnitude matters
- **Manhattan**: Best for high-dimensional sparse data, robust to outliers

### HNSW Configuration

Fine-tune HNSW indexing parameters for optimal performance:

```python
from pyruvector import VectorDB, HNSWConfig

# Custom HNSW configuration
hnsw_config = HNSWConfig(
    m=16,              # Number of bi-directional links (default: 16)
    ef_construction=200,  # Size of dynamic candidate list (default: 200)
    ef_search=50       # Search-time candidate list size (default: 50)
)

db = VectorDB(dimension=128, hnsw_config=hnsw_config)
```

**HNSW Parameters:**
- **m**: Higher values = better recall, more memory (typical: 12-48)
- **ef_construction**: Higher values = better index quality, slower build (typical: 100-500)
- **ef_search**: Higher values = better recall, slower search (typical: 50-200)

### Quantization Configuration

Reduce memory usage with quantization:

```python
from pyruvector import VectorDB, QuantizationConfig, QuantizationType

# Scalar quantization (8-bit) - 4x memory reduction
scalar_config = QuantizationConfig(
    quantization_type=QuantizationType.Scalar,
    bits=8  # 8-bit quantization
)
db = VectorDB(dimension=128, quantization_config=scalar_config)

# Product quantization - even higher compression
product_config = QuantizationConfig(
    quantization_type=QuantizationType.Product,
    num_subvectors=8  # Split into 8 subvectors
)
db = VectorDB(dimension=128, quantization_config=product_config)

# Binary quantization - maximum compression (32x reduction)
binary_config = QuantizationConfig(
    quantization_type=QuantizationType.Binary
)
db = VectorDB(dimension=128, quantization_config=binary_config)
```

**Quantization Types:**
- **None**: No quantization, full precision (default)
- **Scalar**: 8-bit quantization, 4x memory reduction, ~2% accuracy loss
- **Product**: Subvector quantization, 8-32x reduction, configurable accuracy
- **Binary**: 1-bit quantization, 32x reduction, good for semantic similarity

### Full Configuration

Combine all options for maximum control:

```python
from pyruvector import VectorDB, DbOptions, DistanceMetric, HNSWConfig, QuantizationConfig, QuantizationType

# Create comprehensive configuration
options = DbOptions(
    distance_metric=DistanceMetric.Cosine,
    hnsw_config=HNSWConfig(m=32, ef_construction=400, ef_search=100),
    quantization_config=QuantizationConfig(
        quantization_type=QuantizationType.Scalar,
        bits=8
    )
)

# Create database with full configuration
db = VectorDB(dimension=768, options=options)

# Or update existing database
db = VectorDB(dimension=768)
db = db.with_options(options)
```

## CollectionManager - Multi-Tenancy

Manage multiple isolated vector collections with human-readable names:

```python
from pyruvector import CollectionManager, DistanceMetric

# Create collection manager
manager = CollectionManager(base_path="./vector_collections")

# Create collections with different configurations
manager.create_collection(
    name="user_embeddings",
    dimension=384,
    distance_metric=DistanceMetric.Cosine
)

manager.create_collection(
    name="product_images",
    dimension=512,
    distance_metric=DistanceMetric.Euclidean
)

# Create aliases for easy reference
manager.create_alias(collection_name="user_embeddings", alias="users")
manager.create_alias(collection_name="product_images", alias="products")

# Get collection by name or alias
user_db = manager.get_collection("users")
user_db.insert([0.1, 0.2, ...], {"user_id": "123"})

product_db = manager.get_collection("products")
product_db.insert([0.5, 0.3, ...], {"product_id": "456"})

# List all collections
collections = manager.list_collections()
for collection in collections:
    print(f"{collection['name']}: {collection['dimension']}D, {collection['count']} vectors")

# Get collection statistics
stats = manager.get_collection_stats("user_embeddings")
print(f"Count: {stats['count']}, Dimension: {stats['dimension']}")

# Delete collection
manager.delete_collection("old_collection")

# List all aliases
aliases = manager.list_aliases()
print(f"Available aliases: {aliases}")

# Remove alias
manager.remove_alias("users")
```

**CollectionManager Benefits:**
- Isolated namespaces for different use cases
- Centralized management of multiple databases
- Human-readable aliases for developer experience
- Automatic persistence and lifecycle management
- Per-collection configuration and metrics

## API Reference

### VectorDB

#### Constructor

```python
VectorDB(
    dimension: int,
    path: Optional[str] = None,
    distance_metric: Optional[DistanceMetric] = None,
    hnsw_config: Optional[HNSWConfig] = None,
    quantization_config: Optional[QuantizationConfig] = None,
    options: Optional[DbOptions] = None
)
```

Creates a new vector database.

**Parameters:**
- `dimension`: Dimensionality of vectors (must be consistent)
- `path`: Optional file path for persistence
- `distance_metric`: Distance metric to use (default: Cosine)
- `hnsw_config`: HNSW indexing configuration
- `quantization_config`: Vector quantization configuration
- `options`: Complete DbOptions object (overrides individual configs)

#### Methods

##### `insert(vector: List[float], metadata: Optional[Dict] = None) -> int`

Insert a single vector with optional metadata.

Returns the ID of the inserted vector.

```python
vector_id = db.insert([1.0, 2.0, 3.0, 4.0], {"tag": "example"})
```

##### `insert_batch(vectors: List[List[float]], metadatas: Optional[List[Dict]] = None) -> List[int]`

Insert multiple vectors efficiently.

Returns a list of IDs for the inserted vectors.

```python
vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
metadatas = [{"type": "A"}, {"type": "B"}]
ids = db.insert_batch(vectors, metadatas)
```

##### `search(query: List[float], k: int = 10, filter: Optional[Dict] = None) -> List[Dict]`

Search for the k nearest vectors using configured distance metric.

Returns a list of results with `id`, `distance`, `vector`, and `metadata`.

```python
results = db.search([1.0, 0.0, 0.0, 0.0], k=5, filter={"category": {"$eq": "A"}})
```

##### `get(id: int) -> Optional[Dict]`

Retrieve a vector by ID.

Returns a dictionary with `id`, `vector`, and `metadata`, or `None` if not found.

```python
vector_data = db.get(0)
```

##### `delete(id: int) -> bool`

Delete a vector by ID.

Returns `True` if deleted, `False` if not found.

```python
success = db.delete(0)
```

##### `count() -> int`

Get the total number of vectors in the database.

```python
total = db.count()
```

##### `is_empty() -> bool`

Check if the database has no vectors.

```python
if db.is_empty():
    print("Database is empty")
```

##### `contains(id: int) -> bool`

Check if a vector with the given ID exists.

```python
if db.contains(42):
    print("Vector 42 exists")
```

##### `clear()`

Remove all vectors from the database.

```python
db.clear()
```

##### `health() -> Dict`

Get database health status and metrics.

Returns a dictionary with health information, metrics, and warnings.

```python
health = db.health()
print(f"Status: {health['status']}")
print(f"Total vectors: {health['metrics']['total_vectors']}")
print(f"Memory usage: {health['metrics']['memory_bytes']} bytes")
```

##### `with_options(options: DbOptions) -> VectorDB`

Create a new database instance with updated configuration.

```python
new_options = DbOptions(distance_metric=DistanceMetric.Euclidean)
db = db.with_options(new_options)
```

##### `save(path: str)`

Save the database to disk with versioning.

```python
db.save("my_vectors.db")
```

##### `load(path: str) -> VectorDB`

Load a database from disk (class method).

```python
db = VectorDB.load("my_vectors.db")
```

### CollectionManager

#### Constructor

```python
CollectionManager(base_path: str = "./collections")
```

Creates a collection manager for multi-tenancy.

**Parameters:**
- `base_path`: Directory where collections will be stored

#### Methods

##### `create_collection(name: str, dimension: int, **config) -> VectorDB`

Create a new named collection.

**Parameters:**
- `name`: Unique collection name
- `dimension`: Vector dimensionality
- `**config`: Optional VectorDB configuration (distance_metric, hnsw_config, etc.)

Returns the created VectorDB instance.

```python
db = manager.create_collection("embeddings", dimension=768, distance_metric=DistanceMetric.Cosine)
```

##### `get_collection(name: str) -> VectorDB`

Get an existing collection by name or alias.

Returns the VectorDB instance.

```python
db = manager.get_collection("embeddings")
```

##### `delete_collection(name: str) -> bool`

Delete a collection and its data.

Returns `True` if deleted, `False` if not found.

```python
manager.delete_collection("old_embeddings")
```

##### `list_collections() -> List[Dict]`

List all collections with metadata.

Returns a list of dictionaries with `name`, `dimension`, `count`, and `path`.

```python
collections = manager.list_collections()
for col in collections:
    print(f"{col['name']}: {col['count']} vectors")
```

##### `create_alias(collection_name: str, alias: str)`

Create an alias for a collection.

```python
manager.create_alias("user_embeddings", "users")
```

##### `remove_alias(alias: str) -> bool`

Remove an alias.

Returns `True` if removed, `False` if not found.

```python
manager.remove_alias("users")
```

##### `list_aliases() -> Dict[str, str]`

List all aliases.

Returns a dictionary mapping aliases to collection names.

```python
aliases = manager.list_aliases()  # {"users": "user_embeddings"}
```

##### `get_collection_stats(name: str) -> Dict`

Get statistics for a collection.

Returns a dictionary with `count`, `dimension`, and other metadata.

```python
stats = manager.get_collection_stats("embeddings")
```

### Configuration Classes

#### DistanceMetric (Enum)

```python
from pyruvector import DistanceMetric

DistanceMetric.Cosine      # Cosine similarity (default)
DistanceMetric.Euclidean   # L2 distance
DistanceMetric.DotProduct  # Inner product
DistanceMetric.Manhattan   # L1 distance
```

#### HNSWConfig

```python
from pyruvector import HNSWConfig

config = HNSWConfig(
    m=16,                 # Bi-directional links per node (default: 16)
    ef_construction=200,  # Construction-time candidate list (default: 200)
    ef_search=50         # Search-time candidate list (default: 50)
)
```

#### QuantizationConfig

```python
from pyruvector import QuantizationConfig, QuantizationType

# Scalar quantization
config = QuantizationConfig(
    quantization_type=QuantizationType.Scalar,
    bits=8
)

# Product quantization
config = QuantizationConfig(
    quantization_type=QuantizationType.Product,
    num_subvectors=8
)

# Binary quantization
config = QuantizationConfig(
    quantization_type=QuantizationType.Binary
)
```

#### DbOptions

```python
from pyruvector import DbOptions, DistanceMetric, HNSWConfig, QuantizationConfig

options = DbOptions(
    distance_metric=DistanceMetric.Cosine,
    hnsw_config=HNSWConfig(m=32),
    quantization_config=QuantizationConfig(...)
)
```

## Filter Operators

The `filter` parameter in `search()` supports rich query operators:

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equal to | `{"status": {"$eq": "active"}}` |
| `$ne` | Not equal to | `{"status": {"$ne": "deleted"}}` |
| `$gt` | Greater than | `{"score": {"$gt": 0.5}}` |
| `$gte` | Greater than or equal | `{"score": {"$gte": 0.5}}` |
| `$lt` | Less than | `{"age": {"$lt": 30}}` |
| `$lte` | Less than or equal | `{"age": {"$lte": 30}}` |
| `$in` | In array | `{"category": {"$in": ["A", "B"]}}` |
| `$nin` | Not in array | `{"category": {"$nin": ["C", "D"]}}` |
| `$contains` | Array contains value | `{"tags": {"$contains": "important"}}` |
| `$exists` | Field exists | `{"optional_field": {"$exists": true}}` |

### Multiple Conditions

All conditions in a filter must match (AND logic):

```python
filter = {
    "category": {"$eq": "A"},
    "score": {"$gte": 0.7},
    "status": {"$ne": "archived"}
}
results = db.search(query_vector, k=10, filter=filter)
```

## Performance

### HNSW Indexing Performance

Benchmarks with HNSW indexing on consumer hardware (M1 MacBook Pro):

| Operation | Vectors | Distance Metric | Time | Notes |
|-----------|---------|----------------|------|-------|
| Insert (batch) | 10,000 | Cosine | ~50ms | HNSW construction |
| Insert (batch) | 100,000 | Cosine | ~600ms | HNSW construction |
| Search (k=10) | 10,000 | Cosine | ~0.5ms | O(log n) HNSW |
| Search (k=10) | 100,000 | Cosine | ~1.2ms | O(log n) HNSW |
| Search (k=10) | 1,000,000 | Cosine | ~2.5ms | O(log n) HNSW |
| Save to disk | 100,000 | Any | ~200ms | With compression |
| Load from disk | 100,000 | Any | ~150ms | With index rebuild |

### Quantization Impact

| Quantization | Memory Usage | Search Speed | Recall@10 |
|--------------|--------------|--------------|-----------|
| None (FP32) | 100% | 1.0x | 100% |
| Scalar (8-bit) | 25% | 1.8x | 98% |
| Product (8 subvec) | 12.5% | 2.2x | 95% |
| Binary | 3.1% | 3.5x | 90% |

### Distance Metric Performance

| Metric | Relative Speed | Use Case |
|--------|---------------|----------|
| DotProduct | 1.0x (fastest) | Raw embeddings |
| Cosine | 1.1x | Normalized embeddings |
| Euclidean | 1.2x | Spatial data |
| Manhattan | 1.3x | Sparse data |

**Performance Tips:**
- Use HNSW indexing for datasets > 1,000 vectors
- Increase `ef_construction` for better recall (slower build)
- Increase `ef_search` for better recall (slower search)
- Use quantization for large datasets to reduce memory
- Batch insert operations for better throughput
- Use `DotProduct` metric when vectors are pre-normalized

## Development

### Building from Source

```bash
# Install development dependencies
pip install pyruvector[dev]

# Build and install in development mode
maturin develop --release

# Or for faster debug builds
maturin develop
```

### Running Tests

```bash
# Install test dependencies
pip install pyruvector[test]

# Run tests
pytest tests/

# Run with benchmarks
pytest tests/ --benchmark-only
```

### Code Quality

```bash
# Format code
black python/ tests/

# Lint code
ruff python/ tests/

# Type checking
mypy python/
```

### Building Wheels

```bash
# Build wheel for current platform
maturin build --release

# Build for multiple Python versions
maturin build --release --target x86_64-unknown-linux-gnu

# Build with manylinux support
docker run --rm -v $(pwd):/io konstin2/maturin build --release
```

### Project Structure

```
pyruvector/
├── pyruvector/          # Python package
│   ├── __init__.py
│   └── pyruvector.pyi   # Type stubs
├── src/                 # Rust source
│   ├── lib.rs          # Main library
│   ├── vector_db.rs    # Core database
│   ├── hnsw.rs         # HNSW indexing
│   ├── quantization.rs # Quantization
│   ├── distance.rs     # Distance metrics
│   ├── metadata.rs     # Metadata filtering
│   └── collection.rs   # Collection manager
├── examples/            # Usage examples
├── tests/               # Python tests
├── Cargo.toml          # Rust configuration
└── pyproject.toml      # Python configuration
```

## Package Distribution

### PyPI Publication

This package is built using [maturin](https://github.com/PyO3/maturin) and distributed as pre-compiled binary wheels for:
- **Linux**: x86_64, aarch64 (manylinux)
- **macOS**: x86_64, Apple Silicon (M1/M2)
- **Windows**: x86_64

Source distribution (sdist) is also available for custom builds.

### Version Compatibility

| pyruvector | Python | rUvector Ecosystem | Status |
|------------|--------|-------------------|--------|
| 0.1.x      | 3.9+   | 11 crates        | Beta   |

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Format code (`black python/ tests/`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Maintain type hints
- Keep changes focused and atomic

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2024 rUv
Copyright (c) 2025 Rob-otix Ai Ltd

This Rob-otix Ai Ltd version includes enhancements and optimizations for production use.

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/ruvnet/ruvector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ruvnet/ruvector/discussions)
- **Documentation**: [README](https://github.com/ruvnet/ruvector#readme)
- **Rob-otix Ai Ltd**: For enterprise support and custom solutions, contact Rob-otix Ai Ltd
- **Website**: [https://rob-otix.ai](https://rob-otix.ai)

## Roadmap

### In Progress
- [x] HNSW indexing
- [x] Multiple distance metrics
- [x] Quantization (Scalar, Product, Binary)
- [x] Multi-tenancy with CollectionManager
- [x] Distributed clustering
- [x] Graph neural networks
- [x] Snapshot support

### Planned
- [ ] Hybrid search (vector + keyword)
- [ ] GPU acceleration
- [ ] Advanced quantization (PQ+)
- [ ] Real-time index updates
- [ ] Multi-vector queries
- [ ] Vector analytics and visualization
- [ ] Kubernetes operator
- [ ] Cloud-native deployment options

## rUvector Ecosystem

pyruvector provides Python bindings for the complete rUvector Rust ecosystem:

| Crate | Purpose |
|-------|---------|
| `ruvector-core` | Core vector operations, HNSW index, SIMD acceleration |
| `ruvector-collections` | Multi-tenancy, namespace isolation |
| `ruvector-filter` | Advanced metadata filtering |
| `ruvector-metrics` | Prometheus metrics, observability |
| `ruvector-snapshot` | Point-in-time backup and restore |
| `ruvector-graph` | Graph database, Cypher queries, hyperedges |
| `ruvector-gnn` | Graph Neural Networks, adaptive search |
| `ruvector-cluster` | Distributed clustering, coordination |
| `ruvector-raft` | Raft consensus, leader election |
| `ruvector-replication` | Multi-master replication |
| `router-core` + `tiny-dancer-core` | AI routing, FastGRNN inference |

## Acknowledgments

Built with:
- [PyO3](https://github.com/PyO3/pyo3) - Rust bindings for Python
- [maturin](https://github.com/PyO3/maturin) - Build and publish Rust-based Python packages
- [rUvector](https://github.com/ruvnet/ruvector) - High-performance Rust vector database ecosystem
