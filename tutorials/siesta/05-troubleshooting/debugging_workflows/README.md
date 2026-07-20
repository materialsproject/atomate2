# Debugging Workflows

**Category**: troubleshooting/debugging_workflows
**Difficulty**: Intermediate
**Time**: 45 minutes total

## Overview

This section provides systematic debugging strategies for atomate2siesta workflows. Learn how to identify failures, trace job dependencies, and analyze intermediate outputs.

## Available Tutorials

### 01_tracing_job_failures.py

**Tracing Job Failures**

- Identifying which job failed in a flow
- Examining job directories and output files
- Using dry-run mode for testing
- Jobflow-remote debugging commands
- Common debugging patterns
- Comprehensive debugging checklist

**Time**: 20 minutes

### 02_analyzing_intermediate_outputs.py

**Analyzing Intermediate Outputs**

- Understanding SIESTA output files
- Parsing SCF convergence history
- Parsing geometry optimization data
- Plotting convergence (matplotlib)
- Comparing calculated vs reference results
- Reading final structures

**Time**: 25 minutes

## Key Debugging Techniques

### 1. Dry-Run Mode

Always test workflows before running:

```python
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
```

### 2. Job Directory Inspection

```bash
# Find errors
grep -i "error" job_*/siesta.out

# Check convergence
grep "scf:" job_*/siesta.out | tail -20

# Check input parameters
cat job_*/siesta.fdf | head -50
```

### 3. Jobflow-Remote Commands

```bash
# List failed jobs
jf -p PROJECT job list --state FAILED

# Get job details
jf -p PROJECT job info <db_id> --full

# Check output
jf -p PROJECT job output <db_id>
```

## Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| SCF oscillating | Energy goes up/down | Reduce SCF.Mixer.Weight |
| Geometry stuck | Forces not decreasing | Increase MD.NumCGsteps |
| Job not starting | No output files | Check resource allocation |
| Wrong results | Energy differs | Verify pseudopotentials |

## Quick Reference

### Debugging Checklist

- Test with dry_run=True first
- Check structure validity
- Verify pseudopotentials installed
- Read error messages carefully
- Check SCF convergence
- One change at a time

### Useful Patterns

```bash
# Quick error search
grep -i "error" job_*/siesta.out

# SCF history
grep "scf:" job_*/siesta.out

# Energy per step
grep "siesta: E_KS(eV)" job_*/siesta.out
```

## Next Steps

After mastering debugging:

- `../performance_optimization/` - Making calculations faster
- `../common_errors/` - Specific error fixes
- `../../03-advanced-features/04-error-handling/` - Custodian automation

---

*Back to Troubleshooting Index*
