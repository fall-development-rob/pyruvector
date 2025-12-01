"""
Tests for batch operations in VectorDB.
"""

import pytest
import numpy as np
from pyruvector import VectorDB


class TestBatchOperations:
    """Test batch insert, delete, and update operations."""

    def test_batch_insert(self, empty_db, sample_vectors, sample_metadata, dimensions):
        """Test inserting multiple vectors at once."""
        vectors = sample_vectors(dimensions, 10)
        metadata = sample_metadata(10)
        ids = [f"vec_{i}" for i in range(10)]

        # Batch insert
        empty_db.insert_batch(ids=ids, vectors=vectors, metadatas=metadata)

        assert empty_db.stats().vector_count == 10

        # Verify all vectors are searchable
        results = empty_db.search(vectors[0], k=10)
        assert len(results) == 10

    def test_batch_insert_without_metadata(self, empty_db, sample_vectors, dimensions):
        """Test batch insert without metadata."""
        vectors = sample_vectors(dimensions, 5)
        ids = [f"vec_{i}" for i in range(5)]

        empty_db.insert_batch(ids=ids, vectors=vectors, metadatas=[{} for _ in range(5)])

        assert empty_db.stats().vector_count == 5

    def test_batch_insert_mismatched_lengths(self, empty_db, sample_vectors, sample_metadata, dimensions):
        """Test that mismatched vectors/metadata lengths raises error."""
        vectors = sample_vectors(dimensions, 5)
        metadata = sample_metadata(3)  # Fewer metadata items
        ids = [f"vec_{i}" for i in range(5)]

        with pytest.raises(ValueError, match="length"):
            empty_db.insert_batch(ids=ids, vectors=vectors, metadatas=metadata)

    def test_batch_insert_empty_list(self, empty_db):
        """Test batch insert with empty list."""
        empty_db.insert_batch(ids=[], vectors=[], metadatas=[])

        assert empty_db.stats().vector_count == 0

    def test_batch_insert_dimension_validation(self, empty_db, dimensions):
        """Test that batch insert validates all vector dimensions."""
        # Create vectors with mixed dimensions
        vectors = [
            np.random.randn(dimensions).astype(np.float32),
            np.random.randn(dimensions).astype(np.float32),
            np.random.randn(dimensions + 5).astype(np.float32),  # Wrong dimension
        ]
        ids = [f"vec_{i}" for i in range(3)]
        metadatas = [{} for _ in range(3)]

        with pytest.raises(ValueError, match="dimension"):
            empty_db.insert_batch(ids=ids, vectors=vectors, metadatas=metadatas)

    def test_batch_delete(self, empty_db, sample_vectors, dimensions):
        """Test deleting multiple vectors at once."""
        vectors = sample_vectors(dimensions, 10)
        vector_ids = [f"vec_{i}" for i in range(10)]
        metadatas = [{} for _ in range(10)]

        # Insert vectors
        empty_db.insert_batch(ids=vector_ids, vectors=vectors, metadatas=metadatas)

        assert empty_db.stats().vector_count == 10

        # Delete first 5 vectors
        ids_to_delete = vector_ids[:5]
        deleted_count = empty_db.delete_batch(ids=ids_to_delete)

        assert deleted_count == 5
        assert empty_db.stats().vector_count == 5

        # Verify deleted vectors are gone
        results = empty_db.search(vectors[0], k=10)
        result_ids = {r.id for r in results}

        for deleted_id in ids_to_delete:
            assert deleted_id not in result_ids

    def test_batch_delete_empty_list(self, populated_db):
        """Test batch delete with empty list."""
        initial_count = populated_db.stats().vector_count

        deleted_count = populated_db.delete_batch(ids=[])

        assert deleted_count == 0
        assert populated_db.stats().vector_count == initial_count

    def test_batch_delete_nonexistent_ids(self, populated_db):
        """Test batch delete with some nonexistent IDs."""
        initial_count = populated_db.stats().vector_count

        # Mix of nonexistent IDs
        ids_to_delete = ["nonexistent1", "nonexistent2", "nonexistent3"]

        # Should handle gracefully
        deleted_count = populated_db.delete_batch(ids=ids_to_delete)

        # Count should remain same (nothing was deleted)
        assert deleted_count == 0
        assert populated_db.stats().vector_count == initial_count

    def test_large_batch_insert(self, empty_db, sample_vectors, dimensions):
        """Test inserting 1000 vectors in a batch."""
        vectors = sample_vectors(dimensions, 1000)
        ids = [f"vec_{i}" for i in range(1000)]
        metadatas = [{} for _ in range(1000)]

        empty_db.insert_batch(ids=ids, vectors=vectors, metadatas=metadatas)

        assert empty_db.stats().vector_count == 1000

        # Verify searchability
        results = empty_db.search(vectors[0], k=10)
        assert len(results) == 10

    def test_large_batch_with_metadata(self, empty_db, sample_vectors, sample_metadata, dimensions):
        """Test large batch insert with metadata."""
        vectors = sample_vectors(dimensions, 1000)
        metadata = sample_metadata(1000)
        ids = [f"vec_{i}" for i in range(1000)]

        empty_db.insert_batch(ids=ids, vectors=vectors, metadatas=metadata)

        assert empty_db.stats().vector_count == 1000

        # Verify metadata is preserved
        results = empty_db.search(vectors[0], k=5)

        for result in results:
            assert result.metadata is not None
            assert "id" in result.metadata

    def test_batch_insert_performance(self, empty_db, sample_vectors, dimensions):
        """Test that batch insert is more efficient than individual inserts."""
        import time

        vectors = sample_vectors(dimensions, 100)

        # Time individual inserts
        db1 = VectorDB(dimensions=dimensions)
        start_individual = time.time()
        for i, vec in enumerate(vectors):
            db1.insert(f"vec_{i}", vec, {})
        individual_time = time.time() - start_individual
        db1.close()

        # Time batch insert
        ids = [f"vec_{i}" for i in range(100)]
        metadatas = [{} for _ in range(100)]
        start_batch = time.time()
        empty_db.insert_batch(ids=ids, vectors=vectors, metadatas=metadatas)
        batch_time = time.time() - start_batch

        # Batch should be faster (or at least not significantly slower)
        # Allow some margin for overhead
        assert batch_time <= individual_time * 1.5

    def test_batch_update_metadata(self, empty_db, sample_vectors, sample_metadata, dimensions):
        """Test batch updating metadata for multiple vectors."""
        vectors = sample_vectors(dimensions, 5)
        metadata = sample_metadata(5)
        vector_ids = [f"vec_{i}" for i in range(5)]

        # Insert vectors
        empty_db.insert_batch(ids=vector_ids, vectors=vectors, metadatas=metadata)

        # Prepare updated metadata
        updated_metadata = [
            {"id": i, "status": "updated", "version": 2}
            for i in range(5)
        ]

        # Note: batch_update_metadata may not exist in the API
        # This test may need to be skipped or implemented differently
        # For now, we'll delete and re-insert with new metadata
        empty_db.delete_batch(ids=vector_ids)
        empty_db.insert_batch(ids=vector_ids, vectors=vectors, metadatas=updated_metadata)

        # Verify updates
        results = empty_db.search(vectors[0], k=5)

        for result in results:
            assert result.metadata["status"] == "updated"
            assert result.metadata["version"] == 2

    def test_batch_operations_atomicity(self, empty_db, sample_vectors, dimensions):
        """Test that batch operations are atomic (all or nothing)."""
        vectors = sample_vectors(dimensions, 3)

        # Create invalid batch with one wrong dimension
        invalid_vectors = vectors + [np.random.randn(dimensions + 10).astype(np.float32)]
        ids = [f"vec_{i}" for i in range(4)]
        metadatas = [{} for _ in range(4)]

        initial_count = empty_db.stats().vector_count

        with pytest.raises(ValueError):
            empty_db.insert_batch(ids=ids, vectors=invalid_vectors, metadatas=metadatas)

        # Database should remain unchanged
        assert empty_db.stats().vector_count == initial_count

    def test_batch_search(self, populated_db, sample_vectors, dimensions):
        """Test searching with multiple query vectors at once."""
        query_vectors = sample_vectors(dimensions, 3)

        # Batch search - perform individual searches
        all_results = [populated_db.search(query_vec, k=5) for query_vec in query_vectors]

        assert len(all_results) == 3

        for results in all_results:
            assert len(results) <= 5
            for result in results:
                assert hasattr(result, 'id')
                assert hasattr(result, 'score')
                assert hasattr(result, 'metadata')

    def test_mixed_batch_operations(self, empty_db, sample_vectors, dimensions):
        """Test combining multiple batch operations."""
        vectors = sample_vectors(dimensions, 20)

        # Insert first batch
        ids_batch1 = [f"batch1_vec_{i}" for i in range(10)]
        metadatas1 = [{} for _ in range(10)]
        empty_db.insert_batch(ids=ids_batch1, vectors=vectors[:10], metadatas=metadatas1)
        assert empty_db.stats().vector_count == 10

        # Insert second batch
        ids_batch2 = [f"batch2_vec_{i}" for i in range(10)]
        metadatas2 = [{} for _ in range(10)]
        empty_db.insert_batch(ids=ids_batch2, vectors=vectors[10:], metadatas=metadatas2)
        assert empty_db.stats().vector_count == 20

        # Delete half
        deleted_count = empty_db.delete_batch(ids=ids_batch1[:5] + ids_batch2[:5])
        assert deleted_count == 10
        assert empty_db.stats().vector_count == 10
