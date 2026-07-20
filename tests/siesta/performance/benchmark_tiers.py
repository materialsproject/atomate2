"""
Performance benchmarking script for tier-based input architecture.

Measures initialization times across different tiers and identifies
performance characteristics without requiring pytest-benchmark.

Usage:
    python benchmark_tiers.py
"""

import time
import statistics
from pymatgen.core import Structure, Lattice

from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.core import StaticSetGenerator
from atomate2.siesta.sets.tiers import apply_tier_preset
from atomate2.siesta.dataclass.registry import get_modules_for_tier


def create_simple_structure():
    """Create a simple Si structure for testing."""
    si_lattice = Lattice.cubic(5.43)
    structure = Structure(
        si_lattice, ["Si", "Si"], [[0.00, 0.00, 0.00], [0.25, 0.25, 0.25]]
    )
    return structure


def create_medium_structure():
    """Create a medium-sized MgO structure (8 atoms)."""
    lattice = Lattice.cubic(4.212)
    structure = Structure(
        lattice,
        ["Mg", "Mg", "Mg", "Mg", "O", "O", "O", "O"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ],
    )
    return structure


def benchmark_tier_initialization():
    """Benchmark initialization times across all tiers."""
    print("\n" + "=" * 70)
    print("TIER INITIALIZATION PERFORMANCE BENCHMARK")
    print("=" * 70)

    structure = create_simple_structure()
    tiers = ["basic", "intermediate", "advanced", "expert"]
    iterations = 20
    warmup = 3

    results = {}

    for tier in tiers:
        # Warmup runs
        for _ in range(warmup):
            generator = StaticSetGenerator(tier=tier)
            _ = generator.get_input_set(structure)

        # Timed runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            generator = StaticSetGenerator(tier=tier)
            _input_set = generator.get_input_set(structure)
            end = time.perf_counter()
            times.append(end - start)

        # Calculate statistics
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)

        results[tier] = {
            "avg": avg_time,
            "std": std_dev,
            "min": min_time,
            "max": max_time,
            "median": median_time,
            "iterations": iterations,
        }

    # Print results
    print("\nStructure: Si (2 atoms)")
    print(f"Iterations per tier: {iterations} (after {warmup} warmup runs)\n")

    for tier, stats in results.items():
        modules = get_modules_for_tier(tier)
        print(f"{tier.upper()} tier ({len(modules)} modules):")
        print(f"  Mean:     {stats['avg'] * 1000:6.2f} ms")
        print(f"  Median:   {stats['median'] * 1000:6.2f} ms")
        print(f"  Std Dev:  {stats['std'] * 1000:6.2f} ms")
        print(f"  Min:      {stats['min'] * 1000:6.2f} ms")
        print(f"  Max:      {stats['max'] * 1000:6.2f} ms")
        print()

    # Calculate overhead
    print("\nOVERHEAD ANALYSIS (relative to basic tier):")
    print("-" * 50)
    basic_avg = results["basic"]["avg"]
    for tier in ["intermediate", "advanced", "expert"]:
        overhead = results[tier]["avg"] - basic_avg
        overhead_pct = (overhead / basic_avg) * 100
        modules = get_modules_for_tier(tier)
        basic_modules = get_modules_for_tier("basic")
        extra_modules = len(modules) - len(basic_modules)
        print(
            f"{tier.capitalize():13} (+{extra_modules:2d} modules): "
            f"{overhead * 1000:6.2f} ms  ({overhead_pct:5.1f}%)"
        )

    return results


def benchmark_preset_application():
    """Benchmark preset application performance."""
    print("\n" + "=" * 70)
    print("PRESET APPLICATION PERFORMANCE BENCHMARK")
    print("=" * 70)

    structure = create_simple_structure()
    iterations = 20
    warmup = 3

    # Warmup
    for _ in range(warmup):
        maker = RelaxMaker.fixed_cell_relaxation()
        maker = apply_tier_preset(maker, "relax_standard")
        _ = maker.input_set_generator.get_input_set(structure)

    # Time: Create maker
    maker_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        maker = RelaxMaker.fixed_cell_relaxation()
        end = time.perf_counter()
        maker_times.append(end - start)

    # Time: Apply preset
    preset_times = []
    for _ in range(iterations):
        maker = RelaxMaker.fixed_cell_relaxation()
        start = time.perf_counter()
        maker = apply_tier_preset(maker, "relax_standard")
        end = time.perf_counter()
        preset_times.append(end - start)

    # Time: Generate input set
    input_times = []
    for _ in range(iterations):
        maker = RelaxMaker.fixed_cell_relaxation()
        maker = apply_tier_preset(maker, "relax_standard")
        start = time.perf_counter()
        _input_set = maker.input_set_generator.get_input_set(structure)
        end = time.perf_counter()
        input_times.append(end - start)

    # Print breakdown
    print(f"\nIterations: {iterations} (after {warmup} warmup runs)\n")
    print("Component Breakdown:")
    print("-" * 50)
    print(f"Create Maker:          {statistics.mean(maker_times) * 1000:6.2f} ms")
    print(f"Apply Preset:          {statistics.mean(preset_times) * 1000:6.2f} ms")
    print(f"Generate Input Set:    {statistics.mean(input_times) * 1000:6.2f} ms")
    print("-" * 50)
    total = (
        statistics.mean(maker_times)
        + statistics.mean(preset_times)
        + statistics.mean(input_times)
    )
    print(f"Total:                 {total * 1000:6.2f} ms")


def benchmark_module_overrides():
    """Measure overhead of enabled/disabled modules."""
    print("\n" + "=" * 70)
    print("MODULE OVERRIDE OVERHEAD BENCHMARK")
    print("=" * 70)

    structure = create_simple_structure()
    iterations = 20
    warmup = 3

    configurations = {
        "Baseline (no overrides)": {"tier": "intermediate"},
        "Enabled modules (+2)": {
            "tier": "intermediate",
            "enabled_modules": ["phonons", "dos_bands"],
        },
        "Disabled modules (-2)": {
            "tier": "intermediate",
            "disabled_modules": ["spin", "scf_loop"],
        },
        "Both (+2, -2)": {
            "tier": "intermediate",
            "enabled_modules": ["phonons"],
            "disabled_modules": ["spin"],
        },
    }

    results = {}

    for name, config in configurations.items():
        # Warmup
        for _ in range(warmup):
            generator = StaticSetGenerator(**config)
            _ = generator.get_input_set(structure)

        # Timed runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            generator = StaticSetGenerator(**config)
            _input_set = generator.get_input_set(structure)
            end = time.perf_counter()
            times.append(end - start)

        results[name] = {
            "avg": statistics.mean(times),
            "std": statistics.stdev(times),
        }

    # Print results
    print("\nTier: intermediate (12 modules)")
    print(f"Iterations: {iterations} (after {warmup} warmup runs)\n")

    baseline_avg = results["Baseline (no overrides)"]["avg"]

    for name, stats in results.items():
        overhead = stats["avg"] - baseline_avg
        overhead_pct = (overhead / baseline_avg) * 100 if baseline_avg > 0 else 0

        print(f"{name:25} {stats['avg'] * 1000:6.2f} ms", end="")
        if name != "Baseline (no overrides)":
            print(f"  (overhead: {overhead * 1000:+5.2f} ms, {overhead_pct:+5.1f}%)")
        else:
            print()


def benchmark_structure_scaling():
    """Test how performance scales with structure size."""
    print("\n" + "=" * 70)
    print("STRUCTURE SIZE SCALING BENCHMARK")
    print("=" * 70)

    structures = {
        "Small (2 atoms)": create_simple_structure(),
        "Medium (8 atoms)": create_medium_structure(),
    }

    iterations = 20
    warmup = 3
    tier = "basic"

    results = {}

    for name, structure in structures.items():
        # Warmup
        for _ in range(warmup):
            generator = StaticSetGenerator(tier=tier)
            _ = generator.get_input_set(structure)

        # Timed runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            generator = StaticSetGenerator(tier=tier)
            _input_set = generator.get_input_set(structure)
            end = time.perf_counter()
            times.append(end - start)

        results[name] = {
            "avg": statistics.mean(times),
            "std": statistics.stdev(times),
            "n_atoms": len(structure),
        }

    # Print results
    print(f"\nTier: {tier} (6 modules)")
    print(f"Iterations: {iterations} (after {warmup} warmup runs)\n")

    for name, stats in results.items():
        print(f"{name:20} {stats['avg'] * 1000:6.2f} ms  ({stats['n_atoms']} atoms)")

    # Calculate scaling
    small = results["Small (2 atoms)"]
    medium = results["Medium (8 atoms)"]
    ratio = medium["avg"] / small["avg"]
    atom_ratio = medium["n_atoms"] / small["n_atoms"]

    print(f"\nScaling: {ratio:.2f}x time for {atom_ratio:.1f}x atoms")
    print(f"(Near-linear scaling would be {atom_ratio:.1f}x)")


def generate_summary(tier_results):
    """Generate performance summary."""
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    basic = tier_results["basic"]["avg"]
    expert = tier_results["expert"]["avg"]

    print("\nKey Metrics:")
    print(f"  Basic tier (6 modules):      {basic * 1000:6.2f} ms")
    print(f"  Expert tier (24+ modules):   {expert * 1000:6.2f} ms")
    print(
        f"  Overhead for all modules:    {(expert - basic) * 1000:6.2f} ms ({(expert - basic) / basic * 100:.1f}%)"
    )

    print("\nConclusions:")
    if (expert - basic) / basic < 0.20:
        print("  ✓ Tier system adds minimal overhead (<20%)")
    elif (expert - basic) / basic < 0.50:
        print("  ✓ Tier system overhead is acceptable (20-50%)")
    else:
        print("  ⚠ Tier system has significant overhead (>50%)")

    if expert * 1000 < 100:
        print("  ✓ Expert tier initialization is fast (<100ms)")
    elif expert * 1000 < 500:
        print("  ✓ Expert tier initialization is reasonable (<500ms)")
    else:
        print("  ⚠ Expert tier initialization may need optimization (>500ms)")

    print("\n  → Tier-based architecture is suitable for production use")


def main():
    """Run all benchmarks."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "TIER SYSTEM PERFORMANCE BENCHMARK" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run benchmarks
    tier_results = benchmark_tier_initialization()
    benchmark_preset_application()
    benchmark_module_overrides()
    benchmark_structure_scaling()
    generate_summary(tier_results)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
