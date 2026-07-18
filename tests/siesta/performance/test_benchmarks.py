"""
Performance benchmarks for tier-based input architecture.

Measures timing overhead of tier-based initialization compared to baseline,
and identifies any performance bottlenecks in the system.
"""

import pytest
import time
from pymatgen.core import Structure, Lattice

from atomate2.siesta.jobs.core import RelaxMaker
from atomate2.siesta.sets.core import StaticSetGenerator
from atomate2.siesta.sets.tiers import apply_tier_preset
from atomate2.siesta.dataclass.registry import get_modules_for_tier

# The benchmark fixture comes from the optional pytest-benchmark plugin;
# skip this module cleanly when it is not installed.
pytest.importorskip("pytest_benchmark")


@pytest.fixture
def simple_structure():
    """Create a simple Si structure for testing."""
    si_lattice = Lattice.cubic(5.43)
    structure = Structure(
        si_lattice, ["Si", "Si"], [[0.00, 0.00, 0.00], [0.25, 0.25, 0.25]]
    )
    return structure


@pytest.fixture
def medium_structure():
    """Create a medium-sized structure for testing."""
    # MgO conventional cell (8 atoms)
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


class TestTierInitializationPerformance:
    """Benchmark tier-based initialization overhead."""

    def test_basic_tier_initialization_time(self, simple_structure, benchmark):
        """Benchmark basic tier initialization time."""

        def init_basic():
            generator = StaticSetGenerator(tier="basic")
            input_set = generator.get_input_set(simple_structure)
            return input_set

        result = benchmark(init_basic)
        assert result is not None

    def test_intermediate_tier_initialization_time(self, simple_structure, benchmark):
        """Benchmark intermediate tier initialization time."""

        def init_intermediate():
            generator = StaticSetGenerator(tier="intermediate")
            input_set = generator.get_input_set(simple_structure)
            return input_set

        result = benchmark(init_intermediate)
        assert result is not None

    def test_advanced_tier_initialization_time(self, simple_structure, benchmark):
        """Benchmark advanced tier initialization time."""

        def init_advanced():
            generator = StaticSetGenerator(tier="advanced")
            input_set = generator.get_input_set(simple_structure)
            return input_set

        result = benchmark(init_advanced)
        assert result is not None

    def test_expert_tier_initialization_time(self, simple_structure, benchmark):
        """Benchmark expert tier initialization time."""

        def init_expert():
            generator = StaticSetGenerator(tier="expert")
            input_set = generator.get_input_set(simple_structure)
            return input_set

        result = benchmark(init_expert)
        assert result is not None


class TestPresetApplicationPerformance:
    """Benchmark preset application performance."""

    def test_apply_preset_time(self, simple_structure, benchmark):
        """Benchmark time to apply preset to maker."""

        def apply_preset():
            maker = RelaxMaker.fixed_cell_relaxation()
            maker = apply_tier_preset(maker, "relax_standard")
            return maker

        result = benchmark(apply_preset)
        assert result is not None

    def test_preset_with_input_generation(self, simple_structure, benchmark):
        """Benchmark preset application + input generation."""

        def preset_and_generate():
            maker = RelaxMaker.fixed_cell_relaxation()
            maker = apply_tier_preset(maker, "relax_standard")
            input_set = maker.input_set_generator.get_input_set(simple_structure)
            return input_set

        result = benchmark(preset_and_generate)
        assert result is not None


class TestScalingPerformance:
    """Test how performance scales with structure size."""

    def test_basic_tier_small_structure(self, simple_structure, benchmark):
        """Benchmark basic tier with small structure (2 atoms)."""

        def init():
            generator = StaticSetGenerator(tier="basic")
            return generator.get_input_set(simple_structure)

        result = benchmark(init)
        assert result is not None

    def test_basic_tier_medium_structure(self, medium_structure, benchmark):
        """Benchmark basic tier with medium structure (8 atoms)."""

        def init():
            generator = StaticSetGenerator(tier="basic")
            return generator.get_input_set(medium_structure)

        result = benchmark(init)
        assert result is not None


class TestManualTiming:
    """Manual timing tests for detailed analysis."""

    def test_tier_initialization_comparison(self, simple_structure):
        """Compare initialization times across all tiers."""
        tiers = ["basic", "intermediate", "advanced", "expert"]
        timings = {}
        iterations = 10

        for tier in tiers:
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                generator = StaticSetGenerator(tier=tier)
                _input_set = generator.get_input_set(simple_structure)
                end = time.perf_counter()
                times.append(end - start)

            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            timings[tier] = {
                "avg": avg_time,
                "min": min_time,
                "max": max_time,
                "iterations": iterations,
            }

        # Print results
        print("\n" + "=" * 60)
        print("Tier Initialization Performance Comparison")
        print("=" * 60)
        for tier, stats in timings.items():
            modules = get_modules_for_tier(tier)
            print(f"\n{tier.upper()} tier ({len(modules)} modules):")
            print(f"  Average: {stats['avg'] * 1000:.2f} ms")
            print(f"  Min:     {stats['min'] * 1000:.2f} ms")
            print(f"  Max:     {stats['max'] * 1000:.2f} ms")

        # Calculate overhead
        basic_avg = timings["basic"]["avg"]
        for tier in ["intermediate", "advanced", "expert"]:
            overhead = timings[tier]["avg"] - basic_avg
            overhead_pct = (overhead / basic_avg) * 100
            print(
                f"\n{tier.capitalize()} overhead vs. basic: "
                f"{overhead * 1000:.2f} ms ({overhead_pct:.1f}%)"
            )

    def test_preset_application_breakdown(self, simple_structure):
        """Break down preset application into components."""
        iterations = 10

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
            _input_set = maker.input_set_generator.get_input_set(simple_structure)
            end = time.perf_counter()
            input_times.append(end - start)

        # Print breakdown
        print("\n" + "=" * 60)
        print("Preset Application Performance Breakdown")
        print("=" * 60)
        print(
            f"\nCreate Maker:        {sum(maker_times) / len(maker_times) * 1000:.2f} ms"
        )
        print(
            f"Apply Preset:        {sum(preset_times) / len(preset_times) * 1000:.2f} ms"
        )
        print(
            f"Generate Input Set:  {sum(input_times) / len(input_times) * 1000:.2f} ms"
        )
        print(
            f"Total:               {(sum(maker_times) + sum(preset_times) + sum(input_times)) / len(maker_times) * 1000:.2f} ms"
        )

    def test_module_enable_disable_overhead(self, simple_structure):
        """Measure overhead of enabled/disabled modules."""
        iterations = 10

        # Baseline: No overrides
        baseline_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            generator = StaticSetGenerator(tier="intermediate")
            _input_set = generator.get_input_set(simple_structure)
            end = time.perf_counter()
            baseline_times.append(end - start)

        # With enabled modules
        enabled_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            generator = StaticSetGenerator(
                tier="intermediate", enabled_modules=["phonons", "dos_bands"]
            )
            _input_set = generator.get_input_set(simple_structure)
            end = time.perf_counter()
            enabled_times.append(end - start)

        # With disabled modules
        disabled_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            generator = StaticSetGenerator(
                tier="intermediate", disabled_modules=["spin", "scf_loop"]
            )
            _input_set = generator.get_input_set(simple_structure)
            end = time.perf_counter()
            disabled_times.append(end - start)

        # Print results
        baseline_avg = sum(baseline_times) / len(baseline_times)
        enabled_avg = sum(enabled_times) / len(enabled_times)
        disabled_avg = sum(disabled_times) / len(disabled_times)

        print("\n" + "=" * 60)
        print("Module Override Overhead")
        print("=" * 60)
        print(f"\nBaseline (no overrides):     {baseline_avg * 1000:.2f} ms")
        print(f"With enabled modules:        {enabled_avg * 1000:.2f} ms")
        print(f"With disabled modules:       {disabled_avg * 1000:.2f} ms")
        print(
            f"\nEnabled overhead:  {(enabled_avg - baseline_avg) * 1000:.2f} ms "
            f"({(enabled_avg - baseline_avg) / baseline_avg * 100:.1f}%)"
        )
        print(
            f"Disabled overhead: {(disabled_avg - baseline_avg) * 1000:.2f} ms "
            f"({(disabled_avg - baseline_avg) / baseline_avg * 100:.1f}%)"
        )


if __name__ == "__main__":
    # Run benchmarks with pytest-benchmark
    pytest.main([__file__, "-v", "--benchmark-only"])

    # Run manual timing tests
    pytest.main([__file__, "-v", "-k", "Manual", "-s"])
