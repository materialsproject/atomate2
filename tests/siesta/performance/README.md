# Performance Benchmarking for Tier-Based Architecture

This directory contains performance benchmarking tools and results for the tier-based input architecture system.

## Files

- **`benchmark_tiers.py`**: Standalone benchmarking script (no external dependencies beyond standard library)
- **`test_benchmarks.py`**: pytest-based benchmarks (requires `pytest-benchmark`, optional)
- **`BENCHMARK_RESULTS.md`**: Comprehensive benchmark report with analysis

## Quick Start

Run the standalone benchmark script:

```bash
python benchmark_tiers.py
```

This will run all benchmarks and display results in the terminal.

## Benchmarks Included

### 1. Tier Initialization Performance
Measures time to initialize each tier level (basic, intermediate, advanced, expert) and generate input sets.

### 2. Preset Application Performance
Breaks down the time cost of:
- Creating a Maker
- Applying a tier preset
- Generating the input set

### 3. Module Override Overhead
Compares performance when enabling or disabling specific modules.

### 4. Structure Size Scaling
Tests how initialization time scales with structure size (atom count).

## Results Summary

**Key Findings** (see BENCHMARK_RESULTS.md for full details):

- **Basic tier**: ~17 ms
- **Expert tier**: ~23 ms
- **Overhead**: ~35% for all modules (6 ms absolute)
- **Preset application**: < 1 ms
- **Scaling**: Sub-linear with structure size

**Conclusion**: ✅ Production-ready with negligible performance impact

## Running Tests

### Standalone Script (Recommended)
```bash
cd tests/performance
python benchmark_tiers.py
```

Output includes:
- Tier initialization comparison
- Preset application breakdown
- Module override overhead
- Structure scaling analysis
- Performance summary

### pytest-benchmark (Optional)

If you have `pytest-benchmark` installed:

```bash
pip install pytest-benchmark
pytest test_benchmarks.py -v --benchmark-only
```

For manual timing tests:
```bash
pytest test_benchmarks.py -v -k "Manual" -s
```

## Interpreting Results

### Timing Values

- **Mean**: Average time across all iterations
- **Median**: Middle value (robust to outliers)
- **Std Dev**: Variability between runs (< 2ms is excellent)
- **Min/Max**: Range of observed times

### Overhead Analysis

Overhead is calculated relative to the basic tier:
- < 20%: Minimal overhead ✅
- 20-50%: Acceptable overhead ✅
- > 50%: May need optimization ⚠️

Current system: **~35% overhead** for expert tier ✅

### Performance Guidelines

| Use Case                    | Recommended Tier | Time (ms) |
|-----------------------------|------------------|-----------|
| Quick tests                 | basic            | ~17       |
| Most calculations (default) | intermediate     | ~19       |
| Production runs             | advanced/expert  | ~21-23    |

## Customizing Benchmarks

### Adding New Benchmarks

Edit `benchmark_tiers.py` and add a new function:

```python
def benchmark_my_feature():
    """Benchmark a new feature."""
    print("\n" + "=" * 70)
    print("MY CUSTOM BENCHMARK")
    print("=" * 70)

    iterations = 20
    warmup = 3

    # ... your benchmark code ...

    print(f"Results: {avg_time*1000:.2f} ms")
```

Then call it from `main()`:
```python
def main():
    benchmark_tier_initialization()
    benchmark_my_feature()  # Add here
    generate_summary()
```

### Changing Parameters

Modify these constants in benchmark functions:
- `iterations = 20`: Number of timed runs
- `warmup = 3`: Number of warmup runs (not timed)

Increase iterations for more stable statistics, but benchmarks will take longer.

## Continuous Monitoring

For tracking performance over time:

1. **Run benchmarks** before major changes
2. **Save results** with git commit hash
3. **Compare** before/after timings
4. **Flag regressions** > 20% slowdown

Example workflow:
```bash
# Before changes
python benchmark_tiers.py > results_before.txt
git add . && git commit -m "feature: add new module"

# After changes
python benchmark_tiers.py > results_after.txt

# Compare
diff results_before.txt results_after.txt
```

## Performance Tips

### For Users

1. Use presets when available (< 1ms cost)
2. Only enable modules you actually need
3. Tier selection matters little (< 6ms difference)

### For Developers

1. Add new modules with minimal validation logic
2. Use early returns in `setup_*()` methods when possible
3. Avoid expensive operations in module `__init__()`
4. Priority-based initialization is already optimized

## Troubleshooting

### Benchmarks Run Slowly

- Reduce `iterations` from 20 to 5-10
- Remove warmup runs for faster testing
- Comment out specific benchmarks in `main()`

### Inconsistent Timings

- Increase `warmup` runs (3 → 5)
- Close other applications
- Disable CPU throttling/power saving
- Run multiple times and average

### Import Errors

Make sure you're in the correct directory:
```bash
cd /path/to/atomate2siesta
cd tests/performance
python benchmark_tiers.py
```

## Related Documentation

- `tests/TEST_SUMMARY.md`: Unit and integration test coverage
- `tutorials/18-tier-based-calculations/`: User-facing tier tutorial
- `src/atomate2/siesta/dataclass/registry.py`: Module registry implementation
- `src/atomate2/siesta/sets/tiers.py`: Preset system implementation

---

**Last Updated**: 2025-10-08
**Benchmark Version**: 1.0
**Python**: 3.12.0
**Platform**: macOS (darwin)
