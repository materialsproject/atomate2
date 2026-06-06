# Combined Workflow Recipes

Advanced recipes combining multiple calculation types for complex studies.

## Overview

Combined recipes for sophisticated materials discovery and characterization:
- **High-Throughput Screening**: Batch processing and filtering
- **Materials Discovery**: Complete discovery pipeline
- **Publication-Ready**: High-accuracy comprehensive workflows

## Files

- `high_throughput_screening.py` - Batch screening workflow (HIGH priority)
- `materials_discovery.py` - Discovery pipeline (MEDIUM priority)
- `publication_ready_workflow.py` - Publication-quality calculations (MEDIUM priority)
- `README.md` - This file

## Quick Start

```python
from atomate2.siesta.recipes import RecipeBook

# Screen many materials quickly
structures = [struct1, struct2, struct3, ...]
flows = [RecipeBook.high_throughput_screening(s) for s in structures]

# Complete materials discovery
flow = RecipeBook.materials_discovery_pipeline(structures, target_property="band_gap")

# Publication-ready study
flow = RecipeBook.publication_ready_workflow(structure, properties="all")
```

## Workflows

### High-Throughput Screening
1. Quick characterization on all candidates
2. Filter based on criteria
3. Rank by target properties
4. Export top candidates

**Use for**:
- Screening 100s-1000s of materials
- Initial property estimates
- Rapid materials discovery
- Database population

### Materials Discovery Pipeline
1. Initial screening (quick characterization)
2. Filtered detailed study (complete material study)
3. Property-specific optimization
4. Machine learning integration
5. Final validation

**Use for**:
- Discovering new materials
- Property-driven search
- Optimization campaigns
- Research projects

### Publication-Ready Workflow
1. Convergence testing
2. High-accuracy calculations
3. Complete property characterization
4. Automatic figure generation
5. Comprehensive documentation

**Use for**:
- Final calculations for papers
- Experimental comparison
- Benchmark studies
- High-quality data generation

## Screening Criteria Examples

### Band Gap Range
```python
flow = RecipeBook.high_throughput_screening(
    structures,
    filter_criteria={"band_gap": (1.0, 3.0)},  # eV
)
```

### Formation Energy
```python
flow = RecipeBook.high_throughput_screening(
    structures,
    filter_criteria={"formation_energy": (-np.inf, 0.5)},  # eV/atom
)
```

### Multiple Criteria
```python
flow = RecipeBook.high_throughput_screening(
    structures,
    filter_criteria={
        "band_gap": (1.5, 2.5),
        "formation_energy": (-np.inf, 0.3),
        "stability": "stable",
    }
)
```

## Typical Workflows

### Battery Cathode Screening
```python
# Screen 100 candidate cathodes
flows = [RecipeBook.high_throughput_screening(s) for s in candidates]

# Filter: voltage 3-4V, formation energy < 0
filtered = filter_by_criteria(results, voltage=(3, 4), E_form=(-inf, 0))

# Detailed study of top 10
for structure in filtered[:10]:
    flow = RecipeBook.complete_material_study(structure)
```

### Solar Cell Material Discovery
```python
# Target: band gap 1.0-1.6 eV (IR/visible)
flow = RecipeBook.materials_discovery_pipeline(
    structures,
    target_property="band_gap",
    target_range=(1.0, 1.6),
    include_optical=True,
)
```

### Thermoelectric Screening
```python
# Need: semiconductor with low thermal conductivity
flow = RecipeBook.materials_discovery_pipeline(
    structures,
    target_properties=["band_gap", "phonons"],
    filter_criteria={
        "band_gap": (0.5, 2.0),
        "electrical_character": "semiconductor"
    },
    include_thermal=True,
)
```

## Computational Strategy

### Phase 1: Quick Screening
- **Time**: Minutes per structure
- **Accuracy**: ±10-20%
- **Purpose**: Eliminate poor candidates
- **Output**: 10-20% of initial set

### Phase 2: Detailed Study
- **Time**: Hours per structure
- **Accuracy**: Publication quality
- **Purpose**: Full characterization
- **Output**: Top 1-5 materials

### Phase 3: Validation
- **Time**: Days per material
- **Accuracy**: Highest possible
- **Purpose**: Final verification
- **Output**: Experimental candidates

## Time Savings Example

**Traditional approach** (complete study on all):
```
1000 materials × 8 hours = 8000 hours ≈ 1 year
```

**High-throughput approach** (screening + selective detailed):
```
1000 materials × 0.5 hours (quick) = 500 hours
+ 50 materials × 8 hours (detailed) = 400 hours
= 900 hours ≈ 1 month (89% time reduction!)
```

## Best Practices

1. **Define criteria before screening**
2. **Start with quick characterization**
3. **Filter aggressively in early stages**
4. **Use database for result management**
5. **Automate analysis and ranking**
6. **Document decision process**
7. **Validate top candidates thoroughly**

## Integration with Tools

### Databases
```python
# Automatic database storage
from jobflow import run_locally
results = run_locally(flows, store=True)
```

### Machine Learning
```python
# Export for ML training
RecipeBook.export_for_ml(results, format="json")
```

### Visualization
```python
# Generate comparison plots
RecipeBook.plot_screening_results(results, x="formation_energy", y="band_gap")
```

## Next Steps

- **Screening basics**: `02_complete_workflows/quick_characterization.py`
- **Convergence**: `07_convergence_recipes/`
- **Database setup**: `tutorials/04-infrastructure/13-database-storage/`
- **HPC submission**: `tutorials/04-infrastructure/14-job-submission/`
