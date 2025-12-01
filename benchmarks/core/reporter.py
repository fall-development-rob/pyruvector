"""
Benchmark results reporting and formatting utilities.

Supports console tables, JSON export, and markdown report generation.
"""

import json
import time
from typing import Dict, List, Any, Optional, TextIO
from dataclasses import dataclass, asdict
from pathlib import Path
import sys


@dataclass
class BenchmarkResult:
    """Result from a benchmark suite."""

    name: str
    description: str
    timestamp: float
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BenchmarkReporter:
    """
    Formats and exports benchmark results in multiple formats.

    Supports:
    - Console table output with aligned columns
    - JSON export for programmatic analysis
    - Markdown report generation for documentation
    """

    def __init__(self) -> None:
        """Initialize reporter."""
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult) -> None:
        """
        Add a benchmark result.

        Args:
            result: BenchmarkResult to add
        """
        self.results.append(result)

    def clear(self) -> None:
        """Clear all results."""
        self.results.clear()

    def print_table(
        self,
        metrics_to_show: Optional[List[str]] = None,
        file: TextIO = sys.stdout
    ) -> None:
        """
        Print results as a formatted console table.

        Args:
            metrics_to_show: List of metric keys to display (None = all)
            file: Output file stream
        """
        if not self.results:
            print("No benchmark results to display", file=file)
            return

        # Determine metrics to display
        if metrics_to_show is None:
            # Collect all unique metric keys
            all_keys = set()
            for result in self.results:
                all_keys.update(result.metrics.keys())
            metrics_to_show = sorted(all_keys)

        # Build table headers
        headers = ["Benchmark"] + metrics_to_show

        # Build table rows
        rows = []
        for result in self.results:
            row = [result.name]
            for key in metrics_to_show:
                value = result.metrics.get(key, "N/A")
                row.append(self._format_value(value))
            rows.append(row)

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Print table
        self._print_separator(col_widths, file=file)
        self._print_row(headers, col_widths, file=file)
        self._print_separator(col_widths, file=file)

        for row in rows:
            self._print_row(row, col_widths, file=file)

        self._print_separator(col_widths, file=file)

    def export_json(self, filepath: Path) -> None:
        """
        Export results to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        output = {
            "timestamp": time.time(),
            "results": [r.to_dict() for r in self.results]
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

    def export_markdown(self, filepath: Path, title: str = "Benchmark Results") -> None:
        """
        Export results to markdown file.

        Args:
            filepath: Path to output markdown file
            title: Report title
        """
        with open(filepath, 'w') as f:
            # Write header
            f.write(f"# {title}\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Write each result as a section
            for result in self.results:
                f.write(f"## {result.name}\n\n")

                if result.description:
                    f.write(f"{result.description}\n\n")

                # Write metrics table
                f.write("### Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")

                for key, value in sorted(result.metrics.items()):
                    formatted_value = self._format_value(value)
                    f.write(f"| {key} | {formatted_value} |\n")

                f.write("\n")

                # Write metadata if present
                if result.metadata:
                    f.write("### Configuration\n\n")
                    f.write("| Parameter | Value |\n")
                    f.write("|-----------|-------|\n")

                    for key, value in sorted(result.metadata.items()):
                        f.write(f"| {key} | {value} |\n")

                    f.write("\n")

    def print_summary(self, file: TextIO = sys.stdout) -> None:
        """
        Print high-level summary of all results.

        Args:
            file: Output file stream
        """
        if not self.results:
            print("No benchmark results", file=file)
            return

        print("\n" + "=" * 70, file=file)
        print("BENCHMARK SUMMARY", file=file)
        print("=" * 70, file=file)

        for result in self.results:
            print(f"\n{result.name}:", file=file)

            # Print key metrics
            key_metrics = ["mean", "p50", "p95", "p99", "throughput"]
            for key in key_metrics:
                if key in result.metrics:
                    value = self._format_value(result.metrics[key])
                    print(f"  {key:12s}: {value}", file=file)

        print("\n" + "=" * 70 + "\n", file=file)

    def _format_value(self, value: Any) -> str:
        """
        Format metric value for display.

        Args:
            value: Value to format

        Returns:
            Formatted string
        """
        if isinstance(value, float):
            # Format floats with appropriate precision
            if value < 0.01:
                return f"{value:.6f}"
            elif value < 1:
                return f"{value:.4f}"
            elif value < 100:
                return f"{value:.2f}"
            else:
                return f"{value:,.1f}"
        elif isinstance(value, int):
            return f"{value:,}"
        else:
            return str(value)

    def _print_separator(self, col_widths: List[int], file: TextIO) -> None:
        """
        Print table separator line.

        Args:
            col_widths: Width of each column
            file: Output file stream
        """
        parts = ["-" * (w + 2) for w in col_widths]
        print("+" + "+".join(parts) + "+", file=file)

    def _print_row(self, row: List[Any], col_widths: List[int], file: TextIO) -> None:
        """
        Print table row.

        Args:
            row: Row values
            col_widths: Width of each column
            file: Output file stream
        """
        cells = []
        for i, (cell, width) in enumerate(zip(row, col_widths)):
            cell_str = str(cell)
            # Right-align numbers, left-align text
            if i > 0 and isinstance(cell, (int, float)):
                cells.append(f" {cell_str:>{width}} ")
            else:
                cells.append(f" {cell_str:<{width}} ")

        print("|" + "|".join(cells) + "|", file=file)

    def compare_results(
        self,
        baseline_name: str,
        comparison_name: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compare two benchmark results.

        Args:
            baseline_name: Name of baseline benchmark
            comparison_name: Name of comparison benchmark
            metrics: Metrics to compare (None = all common metrics)

        Returns:
            Dictionary mapping metric to percent improvement

        Raises:
            ValueError: If benchmarks not found
        """
        baseline = next((r for r in self.results if r.name == baseline_name), None)
        comparison = next((r for r in self.results if r.name == comparison_name), None)

        if baseline is None:
            raise ValueError(f"Baseline benchmark '{baseline_name}' not found")
        if comparison is None:
            raise ValueError(f"Comparison benchmark '{comparison_name}' not found")

        if metrics is None:
            # Find common metrics
            metrics = list(set(baseline.metrics.keys()) & set(comparison.metrics.keys()))

        improvements = {}
        for metric in metrics:
            if metric not in baseline.metrics or metric not in comparison.metrics:
                continue

            base_val = baseline.metrics[metric]
            comp_val = comparison.metrics[metric]

            if isinstance(base_val, (int, float)) and isinstance(comp_val, (int, float)):
                if base_val != 0:
                    improvement = ((comp_val - base_val) / base_val) * 100
                    improvements[metric] = improvement

        return improvements
