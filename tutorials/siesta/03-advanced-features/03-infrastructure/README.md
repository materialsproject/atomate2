# 04: Infrastructure

**Focus**: Production environment setup for large-scale calculations

**Difficulty**: Intermediate to Advanced

**Prerequisites**:
- Completed basic tutorials ([01-basics](../../01-basics/))
- Access to HPC cluster (for job submission tutorials)
- MongoDB installation (for database tutorials)

---

## Tutorials in This Category

### [01-database-storage](01-database-storage/)
**Description**: Storing calculation results in MongoDB for analysis and retrieval
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~10 min (full setup)
**Key Concepts**: MongoDB integration, jobflow stores, data persistence, querying results

### [02-jobflow-remote](02-jobflow-remote/)
**Description**: Submitting jobs to HPC clusters via jobflow-remote
**Difficulty**: Intermediate
**Time**: ~10 min (setup), varies (execution)
**Key Concepts**: jobflow-remote, HPC integration, queue systems, job management

### [03-error-handling](03-error-handling/)
**Description**: Automatic error detection and recovery with custodian
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~20 min (full example)
**Key Concepts**: Custodian handlers, SCF convergence, error recovery, automatic corrections

### [04-restart-recovery](04-restart-recovery/)
**Description**: SIESTA restart capabilities and calculation recovery
**Difficulty**: Intermediate
**Time**: ~5 min (dry-run), ~15 min (full example)
**Key Concepts**: DM file reuse, restart mechanisms, checkpoint recovery

### [05-dry-run-preview](05-dry-run-preview/)
**Description**: Preview workflows and generate input files without running calculations
**Difficulty**: Beginner
**Time**: ~1 min (instant preview)
**Key Concepts**: Dry-run mode, input file generation, workflow validation, resource planning

---

## Learning Path

For production calculations, we recommend this sequence:

1. **Start**: [05-dry-run-preview](05-dry-run-preview/) - Always preview workflows first!
2. **Data management**: [01-database-storage](01-database-storage/) - Set up result storage
3. **HPC**: [02-jobflow-remote](02-jobflow-remote/) - Submit to cluster
4. **Reliability**: [03-error-handling](03-error-handling/) - Enable automatic error recovery
5. **Advanced**: [04-restart-recovery](04-restart-recovery/) - Handle long calculations

---

## Why Infrastructure Matters

Setting up proper infrastructure:
- ✅ **Saves time**: Automatic error recovery prevents wasted calculations
- ✅ **Organizes data**: Database storage makes results searchable
- ✅ **Scales up**: HPC submission enables large workflows
- ✅ **Reduces errors**: Dry-run previews catch mistakes early
- ✅ **Enables collaboration**: Centralized database for team access

---

## Quick Start Guide

### Step 1: Preview First (Dry-Run)
```python
maker = RelaxMaker(dry_run=True)
job = maker.make(structure)
run_locally(job)
# Generates input files, no SIESTA execution
```

### Step 2: Enable Database Storage
```bash
# Configure database
atomate2siesta-database config --generate

# Test connection
atomate2siesta-database test
```

### Step 3: Submit to HPC
```python
from jobflow_remote import submit_flow

submit_flow(workflow)
# Job submitted to cluster queue
```

### Step 4: Enable Error Handling
```python
maker = RelaxMaker(
    use_custodian=True,  # Enable automatic error recovery
    custodian_max_errors=10
)
```

---

## Database Setup

### MongoDB Installation

**Option 1: Local MongoDB**
```bash
# macOS
brew install mongodb-community
brew services start mongodb-community

# Linux
sudo apt-get install mongodb
sudo systemctl start mongodb
```

**Option 2: Cloud MongoDB (Atlas)**
- Free tier: 512 MB storage
- URL: https://www.mongodb.com/cloud/atlas

### Configuration

Create `~/.jobflow.yaml`:
```yaml
JOB_STORE:
  docs_store:
    type: MongoStore
    database: atomate2siesta
    collection_name: outputs
    host: localhost
    port: 27017
```

---

## HPC Setup

### Jobflow-Remote Installation

```bash
# Install jobflow-remote
pip install jobflow-remote

# Or from development version
atomate2siesta-jobflow-remote install --dev

# Generate configuration
atomate2siesta-jobflow-remote setup
```

### Typical HPC Configuration

```yaml
# Project configuration
project:
  name: my_project

workers:
  - name: my_cluster
    scheduler_type: slurm
    work_dir: /scratch/username/jobflow

resources:
  - name: default
    nodes: 1
    partition: normal
    time: "24:00:00"
```

---

## Error Handling

### Common Error Types

1. **SCF Convergence**: Mixer settings, occupation function
2. **Geometry Optimization**: Step size, convergence criteria
3. **Memory Issues**: Insufficient RAM allocation
4. **Pseudopotential**: Missing or incompatible pseudopotentials

### Custodian Features

- ✅ **Automatic detection**: Parses SIESTA output for errors
- ✅ **Progressive fixes**: Tries multiple correction strategies
- ✅ **Logging**: Complete error history in `custodian.json`
- ✅ **Limits**: Prevents infinite retry loops

---

## Common Issues

### Database Issues

**Issue**: "Connection to MongoDB failed"
**Solution**:
```bash
# Check MongoDB is running
mongo --eval "db.adminCommand('ping')"

# Check ~/.jobflow.yaml configuration
atomate2siesta-database test
```

**Issue**: "Collection not found"
**Solution**:
```bash
# Create database structure
atomate2siesta-database create
```

### HPC Issues

**Issue**: "jobflow-remote command not found"
**Solution**:
```bash
# Install jobflow-remote
atomate2siesta-jobflow-remote install
```

**Issue**: "Job submission failed"
**Solution**:
- Check scheduler type (slurm/pbs/sge)
- Verify work_dir exists and has write permissions
- Test with: `atomate2siesta-jobflow-remote test`

### Error Handling Issues

**Issue**: "Too many errors, calculation aborted"
**Solution**:
```python
# Increase error limit
maker = RelaxMaker(
    use_custodian=True,
    custodian_max_errors=20  # Default: 10
)
```

**Issue**: "Error handler not triggered"
**Solution**:
- Check error is in supported list (see tutorial 03)
- Verify custodian is enabled: `use_custodian=True`

---

## Best Practices

### Dry-Run Workflow
1. **Always start with dry-run**: Preview before running
2. **Check input files**: Inspect generated `siesta.fdf`
3. **Verify structures**: Review structure transformations
4. **Estimate resources**: Calculate job requirements

### Database Management
1. **Regular backups**: Backup MongoDB regularly
2. **Indexing**: Create indexes for frequently queried fields
3. **Cleanup**: Remove old test calculations
4. **Documentation**: Tag calculations with metadata

### HPC Usage
1. **Resource requests**: Don't over-allocate (wastes queue time)
2. **Job arrays**: Use for convergence studies
3. **Monitoring**: Check job status regularly
4. **Results retrieval**: Download results promptly

### Error Handling
1. **Enable by default**: Always use `use_custodian=True` for production
2. **Review logs**: Check `custodian.json` for patterns
3. **Custom handlers**: Add project-specific error handlers if needed
4. **Limits**: Set reasonable `custodian_max_errors` (10-20)

---

## CLI Tools Reference

### Database CLI
```bash
atomate2siesta-database test       # Test connection
atomate2siesta-database create     # Create database
atomate2siesta-database list       # List recent jobs
atomate2siesta-database stats      # Show statistics
atomate2siesta-database clear      # Clear database
```

### Jobflow-Remote CLI
```bash
jf project init                    # Initialize project
jf job submit workflow.py          # Submit workflow
jf job list                        # List jobs
jf job info <job-id>               # Job details
jf runner start                    # Start runner daemon
```

### Cluster Setup CLI
```bash
atomate2siesta-cluster setup       # Set up remote cluster
atomate2siesta-cluster status      # Check cluster status
atomate2siesta-cluster info        # Show documentation
```

---

## Next Category

After setting up infrastructure, proceed to:
- **[03-advanced-workflows](../../02-workflows/)** - Run production workflows with your new setup
- **[05-vibrational-properties](../../02-workflows/06-vibrational-properties/)** - Large-scale phonon calculations
- **[06-surfaces-and-adsorption](../../02-workflows/03-surfaces-and-adsorption/)** - High-throughput surface screening

---

*Back to [Main Tutorial Index](../README.md)*
