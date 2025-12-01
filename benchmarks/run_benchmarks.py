#!/usr/bin/env python3
"""
CLI runner for pyruvector benchmarks.

Usage:
    # Quick smoke test
    python -m benchmarks.run_benchmarks --quick

    # Full benchmark with all databases
    python -m benchmarks.run_benchmarks --db all --vectors 100000 --dimensions 768

    # Compare specific databases
    python -m benchmarks.run_benchmarks --db pyruvector qdrant --vectors 50000

    # Custom output file
    python -m benchmarks.run_benchmarks --output results.json
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Handle both module and script execution
if __name__ == "__main__" and __package__ is None:
    # Running as script
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from benchmarks.compare_vectordbs import run_comparison
else:
    # Running as module
    from .compare_vectordbs import run_comparison


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run pyruvector benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test (small dataset)
  python -m benchmarks.run_benchmarks --quick

  # Benchmark pyruvector only with 100K vectors
  python -m benchmarks.run_benchmarks --db pyruvector --vectors 100000

  # Compare all databases with multiple configurations
  python -m benchmarks.run_benchmarks --db all --vectors 10000 50000 --dimensions 128 384

  # Save results to custom file
  python -m benchmarks.run_benchmarks --output my_results.json
        """,
    )

    parser.add_argument(
        "--db",
        nargs="+",
        default=["pyruvector"],
        choices=["pyruvector", "qdrant", "all"],
        help="Databases to benchmark (default: pyruvector)",
    )

    parser.add_argument(
        "--vectors",
        type=int,
        nargs="+",
        default=[10000],
        help="Vector counts to test (default: 10000)",
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[384],
        help="Vector dimensions to test (default: 384)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[100, 1000],
        help="Batch sizes for insertion (default: 100 1000)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results (default: benchmark_results.json)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke test (1K vectors, 128 dims, batch 100)",
    )

    return parser.parse_args()


def expand_db_list(db_args: List[str]) -> List[str]:
    """Expand 'all' into list of all databases."""
    if "all" in db_args:
        return ["pyruvector", "qdrant"]
    return db_args


def main():
    """Main entry point."""
    args = parse_args()

    # Handle quick mode
    if args.quick:
        print("=" * 80)
        print("QUICK SMOKE TEST MODE")
        print("=" * 80)
        databases = ["pyruvector"]
        vector_counts = [1000]
        dimensions_list = [128]
        batch_sizes = [100]
        output_file = args.output or Path("benchmark_quick.json")
    else:
        databases = expand_db_list(args.db)
        vector_counts = args.vectors
        dimensions_list = args.dimensions
        batch_sizes = args.batch_size
        output_file = args.output or Path("benchmark_results.json")

    # Print configuration
    print("\nBenchmark Configuration:")
    print(f"  Databases: {', '.join(databases)}")
    print(f"  Vector counts: {', '.join(map(str, vector_counts))}")
    print(f"  Dimensions: {', '.join(map(str, dimensions_list))}")
    print(f"  Batch sizes: {', '.join(map(str, batch_sizes))}")
    print(f"  Output file: {output_file}")
    print()

    total_runs = len(databases) * len(vector_counts) * len(dimensions_list) * len(batch_sizes)
    print(f"Total benchmark runs: {total_runs}\n")

    # Confirmation for large benchmarks
    if total_runs > 10 and not args.quick:
        response = input("This will run many benchmarks. Continue? [y/N]: ")
        if response.lower() not in ["y", "yes"]:
            print("Aborted.")
            sys.exit(0)

    # Run benchmarks
    try:
        results = run_comparison(
            databases=databases,
            vector_counts=vector_counts,
            dimensions_list=dimensions_list,
            batch_sizes=batch_sizes,
            output_file=output_file,
        )

        # Summary
        successful = sum(1 for r in results if not r.error)
        failed = sum(1 for r in results if r.error)

        print(f"\n{'='*80}")
        print(f"SUMMARY: {successful} successful, {failed} failed out of {len(results)} total")
        print(f"{'='*80}\n")

        if failed > 0:
            print("Failed benchmarks:")
            for r in results:
                if r.error:
                    print(f"  - {r.db_name}: {r.error}")
            print()

        return 0 if failed == 0 else 1

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running benchmarks: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
