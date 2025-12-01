"""
pyruvector - High-performance Python bindings for ruvector vector database.

Features:
- HNSW indexing with O(log n) search and 95%+ recall
- Multiple distance metrics: Cosine, Euclidean, DotProduct, Manhattan
- Quantization for 4-32x memory compression
- Multi-tenancy with CollectionManager
- Rich metadata filtering

Example:
    from pyruvector import VectorDB, DistanceMetric

    db = VectorDB(dimensions=384, distance_metric=DistanceMetric.cosine())
    db.insert("doc1", [0.1] * 384, {"title": "Example"})
    results = db.search([0.1] * 384, k=5)

    # Or use CollectionManager for multi-tenancy
    from pyruvector import CollectionManager

    manager = CollectionManager()
    manager.create_collection("docs", dimensions=384)
    docs = manager.get_collection("docs")
"""

from ._pyruvector import (
    # Core classes
    VectorDB,
    CollectionManager,
    SearchResult,
    DBStats,

    # Configuration types
    DistanceMetric,
    QuantizationType,
    HNSWConfig,
    QuantizationConfig,
    DbOptions,

    # Stats types
    CollectionStats,
    HealthStatus,

    # Advanced filtering
    PayloadIndexManager,
    IndexType,
    FilterBuilder,
    FilterEvaluator,

    # Metrics
    MetricsRecorder,
    gather_metrics,

    # Snapshot
    SnapshotManager,
    SnapshotInfo,

    # Graph database
    GraphDB,
    Node,
    Edge,
    Hyperedge,
    Transaction,
    IsolationLevel,
    QueryResult,

    # GNN (Graph Neural Networks)
    GNNModel,
    BasicGNNLayer,
    RuvectorLayer,
    GNNConfig,
    PyTrainConfig,
    OptimizerType,
    SchedulerType,
    ReplayBuffer,
    Tensor,
    TrainingMetrics as GNNTrainingMetrics,

    # Cluster/Distributed
    ClusterManager,
    ClusterConfig,
    ClusterNode,
    ClusterStats,
    NodeStatus,
    ShardInfo,
    ShardStatus,
    ReplicaSet,
    Replica,
    ReplicaRole,
    SyncMode,

    # Router/AI Routing
    NeuralRouter,
    RouterConfig,
    Candidate,
    RoutingRequest,
    RoutingResponse,
    RoutingDecision,
    VectorDatabase,
    TrainingDataset,
    TrainingConfig as RouterTrainingConfig,

    # GNN utility functions
    cosine_similarity,
    info_nce_loss,

    # Module functions
    version,
    info,
)

# Aliases for backward compatibility
GNNLayer = BasicGNNLayer
TrainConfig = PyTrainConfig
RoutingMetrics = GNNTrainingMetrics  # Router metrics

__all__ = [
    # Core
    "VectorDB",
    "CollectionManager",
    "SearchResult",
    "DBStats",

    # Configuration
    "DistanceMetric",
    "QuantizationType",
    "HNSWConfig",
    "QuantizationConfig",
    "DbOptions",

    # Stats
    "CollectionStats",
    "HealthStatus",

    # Advanced filtering
    "PayloadIndexManager",
    "IndexType",
    "FilterBuilder",
    "FilterEvaluator",

    # Metrics
    "MetricsRecorder",
    "gather_metrics",

    # Snapshot
    "SnapshotManager",
    "SnapshotInfo",

    # Graph database
    "GraphDB",
    "Node",
    "Edge",
    "Hyperedge",
    "Transaction",
    "IsolationLevel",
    "QueryResult",

    # GNN
    "GNNModel",
    "GNNLayer",
    "BasicGNNLayer",
    "RuvectorLayer",
    "GNNConfig",
    "TrainConfig",
    "PyTrainConfig",
    "GNNTrainingMetrics",
    "OptimizerType",
    "SchedulerType",
    "ReplayBuffer",
    "Tensor",
    "cosine_similarity",
    "info_nce_loss",

    # Cluster
    "ClusterManager",
    "ClusterConfig",
    "ClusterNode",
    "ClusterStats",
    "NodeStatus",
    "ShardInfo",
    "ShardStatus",
    "ReplicaSet",
    "Replica",
    "ReplicaRole",
    "SyncMode",

    # Router
    "NeuralRouter",
    "RouterConfig",
    "Candidate",
    "RoutingRequest",
    "RoutingResponse",
    "RoutingDecision",
    "RoutingMetrics",
    "VectorDatabase",
    "TrainingDataset",
    "RouterTrainingConfig",

    # Functions
    "version",
    "info",
]

__version__ = version()
