# Tutorial: Custodian Error Handling

**Category**: 03-infrastructure
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~15-30 min (local with errors)

---

## Overview

Automatic error detection and recovery for SIESTA calculations using the custodian framework. Enable custodian to handle common calculation failures (SCF convergence, memory issues, walltime limits) with progressive correction strategies.

This tutorial demonstrates how to enable robust, production-ready calculations that can automatically recover from common errors without manual intervention.

---

## What You'll Learn

- Automatic error detection from SIESTA output
- Progressive correction strategies
- Default handlers: SCF, Walltime, Memory, Frozen
- Custom handler configuration
- JSON logging of corrections
- When to enable/disable custodian

---

## Prerequisites

- **Required**: [01-basics](../../../01-basics/) tutorials completed
- **Recommended**: Experience with SCF convergence issues

---

## Key Concepts

### Custodian Framework

**Purpose**: Automatic error detection and recovery for computational workflows

**How it works**:
1. **Monitor**: Watch SIESTA calculation output
2. **Detect**: Identify errors (SCF not converged, memory issues, etc.)
3. **Correct**: Apply progressive fixes (reduce mixing, increase iterations, etc.)
4. **Retry**: Re-run calculation with corrected parameters
5. **Log**: Record all errors and corrections in `custodian.json`

**Philosophy**:
- **Progressive corrections**: Start with minimal changes, escalate if needed
- **Conservative approach**: Preserve user settings when possible
- **Transparent logging**: All actions recorded for review

### Default Handlers

Atomate2SIESTA includes 4 default error handlers:

#### 1. SCFConvergenceHandler

**Detects**: `SCF_NOT_CONVERGED` error (self-consistent field not converging)

**Progressive correction strategy** (5 levels):

**Level 1**: Reduce mixing weight
```python
DM.MixingWeight: 0.1 → 0.05  # More conservative mixing
```

**Level 2**: Increase SCF iterations
```python
MaxSCFIterations: 50 → 100 → 200  # More attempts
```

**Level 3**: Change mixer method
```python
SCF.Mixer.Method: "Pulay" → "Broyden"  # Alternative algorithm
```

**Level 4**: Relax convergence tolerance
```python
DM.Tolerance: 1.0e-4 → 1.0e-3  # Less strict
```

**Level 5**: Add electronic temperature (metals)
```python
OccupationFunction: "FD"
ElectronicTemperature: "300 K"  # Fermi-Dirac smearing
```

**Max attempts**: 5 levels

#### 2. WalltimeHandler

**Detects**: Calculation exceeding walltime limit

**Action**:
- Save current state
- Request calculation restart from last saved state
- Common on HPC systems with job time limits

**Use cases**:
- Long calculations on job queue systems
- Checkpointing large systems
- Multi-day calculations

#### 3. MemoryHandler

**Detects**: Out of memory errors

**Actions**:
- Reduce memory-intensive options
- Adjust buffer sizes
- Suggest increasing available memory

**Note**: Rare with modern systems, but critical for very large calculations

#### 4. FrozenHandler

**Detects**: Calculation not making progress (frozen/hanging)

**Actions**:
- Kill frozen process
- Restart with adjusted settings
- Prevent indefinite hangs

### Correction Logging

All custodian actions are recorded in `custodian.json`:

```json
{
  "errors": [
    {
      "error": "SCF_NOT_CONVERGED",
      "message": "SCF did not converge in 50 iterations",
      "timestamp": "2024-01-15T10:30:22"
    }
  ],
  "actions": [
    {
      "action": "reduce_mixing_weight",
      "old_value": 0.1,
      "new_value": 0.05,
      "level": 1
    }
  ],
  "run_stats": {
    "total_errors": 2,
    "total_corrections": 2,
    "scf_iterations": [45, 67, 42],
    "final_status": "converged"
  }
}
```

**Benefits**:
- Transparency: See exactly what custodian did
- Debugging: Understand why calculations needed corrections
- Optimization: Identify patterns for better initial settings

---

## Configuration Options

### 1. Basic (Default Settings)
```python
EXAMPLE_TYPE = "basic"
```
- Enable custodian with default handlers
- Max errors: 5 (standard)
- All 4 default handlers active
- ~15 minutes runtime
- **Use for**: Standard production calculations

### 2. Custom (Increased Error Tolerance)
```python
EXAMPLE_TYPE = "custom"
```
- Max errors: 10 (increased for difficult systems)
- Conservative initial SCF settings
- More correction attempts allowed
- ~20-30 minutes runtime
- **Use for**: Difficult SCF convergence (metals, magnetic systems)

### 3. Difficult SCF (Test Recovery)
```python
EXAMPLE_TYPE = "difficult_scf"
```
- Intentionally difficult initial settings
- Tests custodian's recovery capabilities
- Max errors: 8
- ~20-30 minutes runtime
- **Use for**: Understanding custodian behavior, testing robustness

### 4. Geometry Convergence
```python
MODE = 1  # or 2, 3, 4
```
- Four modes for geometry convergence handling
- MODE 1: Fast/dirty (lenient, no custodian)
- MODE 2: Auto-recovery (custodian, lenient validator)
- MODE 3: Strict checking (no custodian, strict validator)
- MODE 4: Paranoid (custodian + strict validator)
- **Use for**: Understanding geometry convergence handling, choosing right mode

---

## Quick Start

```bash
# 1. Preview workflow structure
# Edit tutorial.py: EXAMPLE_TYPE = "basic", RUN_MODE = "dry_run"
python tutorial.py

# 2. Run with custodian (local)
# Edit tutorial.py: RUN_MODE = "local"
python tutorial.py

# 3. Check corrections log
cat job_*/custodian.json | jq
```

---

## Expected Output

### Successful Calculation (No Errors)

```
================================================================================
EXECUTION
================================================================================

Mode: LOCAL

Running with custodian...
Watch for error detection and corrections

✅ Calculation complete!

📊 Check custodian.json for corrections:
   $ cat job_*/custodian.json | jq
```

**custodian.json** (no errors):
```json
{
  "errors": [],
  "actions": [],
  "run_stats": {
    "total_errors": 0,
    "scf_iterations": [42],
    "final_status": "converged"
  }
}
```

### Calculation with SCF Errors

**Console output**:
```
Running with custodian...

⚠️  Error detected: SCF_NOT_CONVERGED
🔧 Applying correction: Reduce mixing weight (0.1 → 0.05)
♻️  Restarting calculation...

⚠️  Error detected: SCF_NOT_CONVERGED
🔧 Applying correction: Increase MaxSCFIterations (50 → 100)
♻️  Restarting calculation...

✅ Calculation converged!
```

**custodian.json** (with corrections):
```json
{
  "errors": [
    {
      "error": "SCF_NOT_CONVERGED",
      "message": "SCF did not converge in 50 iterations",
      "scf_iterations": 50
    },
    {
      "error": "SCF_NOT_CONVERGED",
      "message": "SCF did not converge in 100 iterations",
      "scf_iterations": 100
    }
  ],
  "actions": [
    {
      "action": "reduce_mixing_weight",
      "level": 1,
      "old_value": 0.1,
      "new_value": 0.05
    },
    {
      "action": "increase_scf_iterations",
      "level": 2,
      "old_value": 50,
      "new_value": 100
    }
  ],
  "run_stats": {
    "total_errors": 2,
    "total_corrections": 2,
    "scf_iterations": [50, 100, 78],
    "final_status": "converged"
  }
}
```

---

## When to Use Custodian

### ✅ Enable Custodian For:

**Production Calculations**:
- Important calculations where robustness > speed
- Want unattended completion overnight/weekend
- Can't afford manual intervention

**Batch Workflows**:
- Large-scale screening (100+ structures)
- Automated workflows
- High-throughput calculations

**HPC Job Submission**:
- Limited job queue access
- Expensive to re-submit failed jobs
- Walltime limits require checkpointing

**Difficult Systems**:
- Metals (Fermi surface)
- Magnetic systems (spin convergence)
- Systems with history of SCF issues
- Unknown/untested structures

**Automated Workflows**:
- No manual monitoring possible
- Need guaranteed completion
- Production pipelines

### ❌ Disable Custodian For:

**Debugging**:
- Understanding WHY calculation fails
- Testing new parameters
- Learning about failure modes

**Parameter Testing**:
- Want to see raw results without corrections
- Testing convergence systematically
- Comparing different settings

**Quick Tests**:
- Dry-run mode (custodian not needed)
- Structure validation
- Quick energy estimates

**Known Good Settings**:
- Well-tested parameters for your system
- Calculations that always work
- When speed > robustness

---

## Settings Guidelines

### Standard Production

**Configuration**:
```python
use_custodian = True
custodian_max_errors = 5  # Standard
```

**Use for**:
- General production calculations
- Most materials
- Typical convergence behavior

**Expected**: 0-2 corrections for well-behaved systems

### Difficult Convergence

**Configuration**:
```python
use_custodian = True
custodian_max_errors = 10  # Increased tolerance
user_params = {
    "MaxSCFIterations": 100,   # Start higher
    "DM.MixingWeight": 0.05,    # Conservative mixing
}
```

**Use for**:
- Metals
- Magnetic systems
- Large systems (>100 atoms)
- Complex structures (surfaces, interfaces)

**Expected**: 3-8 corrections possible

### Very Difficult Systems

**Configuration**:
```python
use_custodian = True
custodian_max_errors = 15  # Maximum tolerance (rare)
user_params = {
    "MaxSCFIterations": 200,
    "DM.MixingWeight": 0.01,
    "OccupationFunction": "MP",
    "ElectronicTemperature": "300 K",
}
```

**Use for**:
- Pathologically difficult systems
- When all else fails
- Research into difficult convergence

**Note**: If needing >15 corrections, consider:
- Is structure reasonable?
- Are parameters appropriate?
- Is SIESTA suitable for this system?

---

## Common Scenarios

### Scenario 1: Simple Insulator

**System**: Silicon, NaCl, Diamond

**Expected behavior**:
- Usually no errors
- custodian.json shows `"errors": []`
- Completes in first run

**Settings**: Standard (max_errors=5)

### Scenario 2: Metal

**System**: Al, Cu, Au

**Initial error**: SCF_NOT_CONVERGED

**Custodian actions**:
1. Reduce mixing weight
2. Add Fermi smearing (OccupationFunction: MP)
3. Increase MaxSCFIterations

**Final result**: Converged in 2-3 corrections

**Settings**: Standard (max_errors=5)

### Scenario 3: Magnetic System

**System**: Fe, Ni, magnetic oxides

**Initial errors**: Multiple SCF_NOT_CONVERGED

**Custodian actions**:
1. Reduce mixing weight
2. Increase SCF iterations
3. Change mixer method
4. Add electronic temperature
5. Relax DM.Tolerance

**Final result**: Converged in 4-6 corrections

**Settings**: Difficult (max_errors=10)

### Scenario 4: Surface/Slab

**System**: Metal surface with adsorbate

**Initial errors**: SCF_NOT_CONVERGED

**Custodian actions**:
1. Reduce mixing weight
2. Increase SCF iterations
3. Add electronic temperature (if metal)

**Final result**: Converged in 2-4 corrections

**Settings**: Difficult (max_errors=10)

---

## Best Practices

### Workflow Strategy

**1. Start Conservative**:
```python
# Good initial settings
user_params = {
    "MaxSCFIterations": 100,  # Not too low
    "DM.MixingWeight": 0.1,    # Moderate
    "DM.Tolerance": "1.0e-4",  # Standard
}
```

**2. Enable Custodian**:
```python
use_custodian = True
custodian_max_errors = 5  # Standard first
```

**3. Review Corrections**:
```bash
# Check what custodian did
cat job_*/custodian.json | jq '.errors'
cat job_*/custodian.json | jq '.actions'
```

**4. Optimize Initial Settings**:
- If many corrections: Incorporate successful corrections into initial settings
- If no corrections: Can try looser settings for speed
- Document successful parameter sets

### Production Checklist

✅ **Enable custodian** for production calculations

✅ **Set appropriate max_errors**:
- Standard systems: 5
- Difficult systems: 10
- Very difficult: 15 (rare)

✅ **Review custodian.json** after completion:
- What errors occurred?
- What corrections worked?
- Can initial parameters be improved?

✅ **Document corrections** for future calculations:
- Record successful parameter sets
- Build library of working settings
- Share with team/community

✅ **Monitor patterns**:
- Same errors repeating?
- Certain materials always need corrections?
- Update default settings accordingly

### Performance Considerations

**Overhead**:
- Custodian monitoring: Negligible (<1%)
- Error detection: ~1-2 seconds per check
- Restart overhead: Minimal (continues from last state)

**Time savings**:
- Avoids manual restarts: Hours to days
- Prevents job queue re-submissions: Days
- Enables overnight/weekend runs: Priceless

**Trade-off**:
- Slightly slower individual runs (corrections)
- Much faster overall workflow (no manual intervention)
- **Recommendation**: Enable for all production work

---

## Advanced Topics

### Custom Handlers

For specialized error handling, write custom handlers:

```python
from atomate2.siesta.custodian.handlers import ErrorHandler

class MyCustomHandler(ErrorHandler):
    """Custom error handler for specific error type."""

    def check(self, dir_path: str) -> bool:
        """Check if error occurred."""
        # Parse SIESTA output
        # Return True if error detected
        pass

    def correct(self, dir_path: str) -> dict:
        """Apply correction."""
        # Modify input files
        # Return correction metadata
        pass
```

**Use cases**:
- System-specific errors
- Custom convergence strategies
- Integration with other codes
- Specialized workflows

**Documentation**: See `src/atomate2/siesta/custodian/handlers.py` for examples

### Handler Priority

Handlers are executed in order:

1. **SCFConvergenceHandler** (most common)
2. **WalltimeHandler** (HPC-specific)
3. **MemoryHandler** (resource errors)
4. **FrozenHandler** (last resort)

Custom handlers can be inserted at any position:

```python
from atomate2.siesta.custodian import DEFAULT_HANDLERS

custom_handlers = [
    MyCustomHandler(),
] + DEFAULT_HANDLERS  # Custom handler runs first
```

### Multi-Stage Corrections

For complex workflows, custodian can handle multi-stage corrections:

**Example**: Phonon calculation with tight forces
```python
# Stage 1: Relax structure (moderate settings)
relax_maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,
    custodian_max_errors=5,
)

# Stage 2: Force calculations (tight settings)
phonon_maker = SiestaPhononMaker(
    use_custodian=True,
    custodian_max_errors=10,  # More tolerance for forces
    user_params={
        "Mesh.Cutoff": "300 Ry",  # Higher than relax
        "kpts": [6, 6, 6],         # Denser than relax
    }
)
```

**Benefit**: Each stage gets appropriate error tolerance

---

## Troubleshooting

### Issue 1: Too Many Errors

**Symptoms**: Calculation exceeds `custodian_max_errors`

**Possible causes**:
1. Structure is unreasonable (overlapping atoms, wrong lattice)
2. Parameters are inappropriate for system
3. System is genuinely very difficult

**Solutions**:
1. **Check structure**:
   ```python
   from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
   sga = SpacegroupAnalyzer(structure)
   print(sga.get_space_group_symbol())  # Reasonable?
   print(structure.distance_matrix.min())  # Atoms too close?
   ```

2. **Try stricter initial settings**:
   ```python
   user_params = {
       "MaxSCFIterations": 200,
       "DM.MixingWeight": 0.01,  # Very conservative
       "DM.Tolerance": "1.0e-3",  # Relaxed
   }
   ```

3. **Increase max_errors** (up to 15):
   ```python
   custodian_max_errors = 15
   ```

4. **Check if SIESTA is appropriate**:
   - Very heavy elements? Consider other codes
   - Strong correlation? SIESTA may struggle
   - Review literature for your system

### Issue 2: Custodian Not Running

**Symptoms**: No `custodian.json` file created

**Causes**:
1. Custodian only runs with actual SIESTA (not dry_run)
2. `use_custodian=False` in settings
3. Job failed before custodian started

**Solutions**:
1. **Check RUN_MODE**:
   ```python
   RUN_MODE = "local"  # Not "dry_run"
   ```

2. **Verify custodian enabled**:
   ```python
   print(f"Custodian enabled: {maker.use_custodian}")  # Should be True
   ```

3. **Check logs**:
   ```bash
   cat job_*/log.json  # SIESTA ran?
   ls job_*/           # Files present?
   ```

### Issue 3: Corrections Not Effective

**Symptoms**: Same error repeating despite corrections

**Causes**:
1. Correction strategy not suitable for this error type
2. System needs different approach
3. Convergence impossible with current settings

**Solutions**:
1. **Review custodian.json**:
   ```bash
   cat job_*/custodian.json | jq '.actions'
   # What was tried?
   ```

2. **Try manual corrections**:
   - Identify pattern in corrections
   - Apply more aggressive fixes manually
   - Test if convergence possible

3. **Consider alternatives**:
   - Different functional (GGA → LDA)
   - Different basis set (DZP → SZ)
   - Different pseudopotentials
   - Different DFT code

### Issue 4: Slow Progress

**Symptoms**: Many corrections, slow convergence

**Cause**: Progressive corrections starting from very conservative

**Solutions**:
1. **Start with better settings**:
   ```python
   # Don't start with intentionally difficult settings
   user_params = {
       "MaxSCFIterations": 100,  # Not 20
       "DM.MixingWeight": 0.1,    # Not 0.5
   }
   ```

2. **Analyze previous corrections**:
   - What final settings worked?
   - Use those as initial settings next time

3. **Build parameter library**:
   - Document successful settings per material type
   - Share within research group

---

## Tips for Success

✅ **Enable for production**: Always use custodian for important calculations

✅ **Review logs**: Check `custodian.json` after every run

✅ **Document patterns**: Build library of successful parameter sets

✅ **Start conservative**: Better initial settings = fewer corrections

✅ **Appropriate max_errors**: Standard (5), difficult (10), very difficult (15)

✅ **Monitor overhead**: Custodian adds minimal time (<1%)

✅ **Test without custodian first**: Understand failure modes before automating

✅ **Share knowledge**: Document working settings for your materials

---

## Integration with Workflows

### With Database Storage

Custodian corrections are automatically saved in database:

```python
# Enable both custodian and database
maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,
    custodian_max_errors=5,
)

job = maker.make(structure)
results = run_locally(job, create_folders=True)

# custodian.json stored in task document
# Query later:
# db.tasks.find_one({"custodian.errors": {"$ne": []}})
```

### With Jobflow Remote (HPC)

Custodian runs automatically on remote systems:

```bash
# Submit job
atomate2siesta-jobflow-remote test

# Custodian runs on cluster
# Check results when complete:
jf job info <job_id>
jf job output <job_id>  # See custodian.json
```

**Benefits**:
- No manual intervention on cluster
- Automatic recovery from errors
- Logged corrections for review

### With Convergence Studies

Enable custodian for all convergence calculations:

```python
from atomate2.siesta.flows.convergence import KpointsConvergenceMaker

workflow = KpointsConvergenceMaker(
    kpoints_list=[[2,2,2], [4,4,4], [6,6,6], ...],
    use_custodian=True,  # Each calculation protected
    custodian_max_errors=5,
)
```

**Benefit**: Entire convergence study completes even if some calculations difficult

---

## Geometry Convergence Handling (NEW!)

Automatic geometry convergence recovery and flexible validation modes.

### The Problem

Non-converged relaxations can silently succeed, leading to:
- Bad geometries in database
- Incorrect downstream calculations (phonons, NEB)
- No warning or error

**User question that triggered this**: "What happens if the Relaxation is not converged?!"

### The Solution: Dual-Mode System

#### 1. GeometryConvergenceHandler (Automatic Recovery)

Progressive MD step increases when geometry doesn't converge:

**Level 1**: +50% steps (200 → 300)
**Level 2**: +100% steps (200 → 400)
**Level 3**: +150% + FIRE method
**Level 4**: +200% + relaxed tolerance
**Level 5**: Max 1000 steps + Broyden optimizer

Similar to SCFRelaxationHandler but for geometry convergence.

#### 2. Strict Convergence Mode (Quality Gate)

New parameter: `strict_convergence: bool = False`

**Lenient mode** (default):
- Allow non-converged geometries
- Use for dirty/fast calculations
- Handler can fix issues

**Strict mode** (`strict_convergence=True`):
- Must converge or validation fails
- Clear error messages with suggestions
- Use for production with quality assurance

#### 3. Convergence Metadata (Always Tracked)

New fields in OutputDoc (all modes):
- `geometry_converged: bool` - Did it converge?
- `final_max_force: float` - Max force (eV/Ang)
- `force_tolerance: float` - Tolerance used (eV/Ang)

### Four Usage Modes

| Mode | Custodian | Strict | Use Case |
|------|-----------|--------|----------|
| 1 | False | False | Fast/dirty (screening) |
| 2 | True | False | Production with auto-recovery |
| 3 | False | True | Production with strict checking |
| 4 | True | True | Paranoid (guaranteed convergence) |

**Example: Mode 2 (Recommended for Production)**
```python
maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,  # Enable auto-recovery
    strict_convergence=False,  # Lenient (default)
)
# GeometryConvergenceHandler will automatically:
# - Detect non-converged geometry
# - Increase MD.NumCGsteps progressively
# - Try alternative methods (FIRE, Broyden)
# - Retry until converged or max attempts
```

**Example: Mode 4 (Critical Calculations)**
```python
maker = RelaxMaker.fixed_cell_relaxation(
    use_custodian=True,  # Enable auto-recovery
    strict_convergence=True,  # Enforce quality
)
# Behavior:
# - Handler tries to fix (up to 5 attempts)
# - Validator enforces quality after corrections
# - Result: Either converged OR clear error
# Perfect for phonon/NEB workflows!
```

### Tutorial: 04_geometry_convergence.py

Run all 4 modes to see different behaviors:

```bash
# Try Mode 1: Fast/dirty
# Edit: MODE = 1
python 04_geometry_convergence.py

# Try Mode 2: Auto-recovery
# Edit: MODE = 2
python 04_geometry_convergence.py

# Try Mode 3: Strict checking
# Edit: MODE = 3
python 04_geometry_convergence.py

# Try Mode 4: Paranoid
# Edit: MODE = 4
python 04_geometry_convergence.py
```

### When to Use Each Mode

**Mode 1 (Default)**: Research, screening
- Fast calculations more important than convergence
- Testing workflows
- Pre-relaxation before fine relaxation

**Mode 2 (Production)**: Standard production
- Unattended calculations
- HPC job submission
- Want automatic recovery

**Mode 3 (Strict)**: Production with immediate feedback
- Need to know convergence status immediately
- Want clear error messages
- Testing if convergence possible

**Mode 4 (Critical)**: Phonon, NEB, critical calculations
- Require guaranteed converged geometries
- Maximum robustness + quality
- Can't accept non-converged results

---

## Next Steps

After completing this tutorial:

1. **Run Basic Example**:
   → Test custodian with simple system
   → Review `custodian.json` output
   → Understand default behavior

2. **Test Difficult System**:
   → Run `EXAMPLE_TYPE = "difficult_scf"`
   → See progressive corrections in action
   → Understand recovery capabilities

3. **Try Geometry Convergence Modes** (NEW!):
   → Run `04_geometry_convergence.py` with MODE = 1, 2, 3, 4
   → Compare behavior for each mode
   → Check convergence metadata in results
   → Choose appropriate mode for your workflows

4. **Apply to Your Work**:
   → Enable custodian for production calculations
   → Document corrections for your systems
   → Build parameter library
   → Use appropriate geometry convergence mode

5. **Advanced Topics**:
   → Write custom handlers if needed
   → Integrate with workflows
   → Optimize initial parameters based on corrections

6. **Related Tutorials**:
   → [Database Storage](../01-database-storage/) - Store custodian logs
   → [Jobflow Remote](../02-jobflow-remote/) - HPC submission with custodian
   → [Convergence Studies](../../../02-workflows/01-convergence/) - Custodian + convergence

---

## References

1. **Custodian Documentation**: MaterialsProject custodian library
   - https://github.com/materialsproject/custodian
   - General framework for error handling

2. **SIESTA Error Handling**: Common SIESTA errors and solutions
   - SIESTA manual: Error messages section
   - Community forums: siesta.icmab.es

3. **Atomate2SIESTA Handlers**: Implementation details
   - `src/atomate2/siesta/custodian/handlers.py`
   - Source code for all handlers

4. **Online Documentation**:
   - See https://github.com/materialsproject/atomate2 for technical details

---

*Back to [03-infrastructure](../README.md) | [Main Tutorial Index](../../README.md)*
