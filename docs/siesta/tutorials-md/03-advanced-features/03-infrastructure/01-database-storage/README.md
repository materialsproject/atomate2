# Database Storage Tutorial

## Overview

This tutorial demonstrates MongoDB integration for centralizing calculation results, enabling easy querying, workflow provenance tracking, and collaboration. The **atomate2siesta-database CLI** provides convenient database management tools.

**Why use database storage?**
- ✅ **Centralized**: All results in one place
- ✅ **Queryable**: Search by formula, directory, parameters
- ✅ **Scalable**: Handle thousands of calculations
- ✅ **Shareable**: Team collaboration
- ✅ **Provenance**: Complete workflow tracking

---

## Quick Start

### 1. Install MongoDB

```bash
# macOS
brew install mongodb-community
brew services start mongodb-community

# Ubuntu/Debian
sudo apt install mongodb
sudo systemctl start mongodb

# Verify MongoDB is running
mongosh --eval "db.version()"
```

### 2. Install Python Packages

```bash
pip install pymongo maggma
```

### 3. Generate Configuration

```bash
# Create ~/.jobflow.yaml automatically
atomate2siesta-database config --generate

# Or manually create ~/.jobflow.yaml:
```

```yaml
JOB_STORE:
  docs_store:
    type: MongoStore
    database: atomate2siesta
    collection_name: tasks
    host: localhost
    port: 27017
```

### 4. Test Connection

```bash
atomate2siesta-database test
```

---

## Database CLI Commands

### Essential Commands

| Command | Description | Example |
|---------|-------------|---------|
| `config --generate` | Generate ~/.jobflow.yaml | `atomate2siesta-database config --generate` |
| `test` | Test MongoDB connection | `atomate2siesta-database test` |
| `create` | Create database/collection | `atomate2siesta-database create` |
| `list` | List recent documents | `atomate2siesta-database list --limit 10` |
| `query <formula>` | Query by formula | `atomate2siesta-database query Si` |
| `stats` | Show statistics | `atomate2siesta-database stats` |
| `clear` | Clear collection | `atomate2siesta-database clear --force` |

### Custom MongoDB Settings

All commands support custom connection parameters:

```bash
atomate2siesta-database test \
    --host myserver.com \
    --port 27018 \
    --database my_calculations \
    --collection my_tasks
```

### Examples

```bash
# Test connection
atomate2siesta-database test

# List 5 most recent calculations
atomate2siesta-database list --limit 5

# Find all Silicon calculations
atomate2siesta-database query Si

# Show database statistics
atomate2siesta-database stats

# DANGER: Clear all documents
atomate2siesta-database clear --force
```

---

## Running Calculations with Database Storage

### Basic Workflow

```python
from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

# Load structure
structure = Structure.from_file("Si.cif")

# Create relaxation job
relax_maker = RelaxMaker.fixed_cell_relaxation()
job = relax_maker.make(structure)

# Run with database storage (uses ~/.jobflow.yaml)
results = run_locally(job, create_folders=True)

# Results automatically stored in MongoDB!
```

**What gets stored?**
- Complete `SiestaTaskDoc` with structure, parameters, outputs
- Energy, forces, stress
- SIESTA input/output files
- Calculation metadata (directory, task_id, timestamps)
- Workflow provenance

### Explicit JobStore (Without jobflow.yaml)

If you prefer not to use `~/.jobflow.yaml`, you can define the JobStore explicitly in your scripts:

```python
from jobflow import SETTINGS, JobStore, run_locally
from maggma.stores import MongoStore

# Define MongoDB store
store = MongoStore(
    database="atomate2siesta",
    collection_name="tasks",
    host="localhost",
    port=27017,
)

# Create JobStore and set in SETTINGS
job_store = JobStore(docs_store=store)
SETTINGS.JOB_STORE = job_store

# Run job (uses SETTINGS.JOB_STORE)
results = run_locally(job, create_folders=True)
```

**See**: `09_explicit_jobstore.py` for complete example

### Dry-Run First (Recommended)

```python
# Validate setup before running expensive calculation
relax_maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    dry_run_output_dir="preview"
)

job = relax_maker.make(structure)
run_locally(job)  # Quick preview, no database storage

# Inspect generated files
# cat preview/*/siesta.fdf

# Then run real calculation (remove dry_run=True)
```

---

## Querying Database Results

### Tutorial Scripts

We provide several ready-to-use Python scripts:

| Script | Description | Usage |
|--------|-------------|-------|
| `02_store_calculation.py` | Basic database storage | Store calculation with jobflow.yaml |
| `09_explicit_jobstore.py` | Explicit JobStore definition | Store without jobflow.yaml |
| `03_query_results.py` | Simple query example | Basic document retrieval |
| `04_query_by_formula.py` | Search by chemical formula | Find all Si, MgO, etc. calculations |
| `05_query_by_energy.py` | Filter by energy range | Find low-energy structures |
| `06_query_bandgap.py` | Classify materials | Metals, semiconductors, insulators |
| `07_query_recent.py` | Most recent calculations | Latest 10 calculations |
| `08_query_statistics.py` | Database statistics | Total counts, averages, distributions |

**Run any script directly:**

```bash
cd tutorials/03-advanced-features/03-infrastructure/01-database-storage
python3 02_store_calculation.py  # Store with jobflow.yaml
python3 09_explicit_jobstore.py  # Store without jobflow.yaml
python3 03_query_results.py      # Simple query
python3 04_query_by_formula.py   # Query by formula
python3 05_query_by_energy.py    # Query by energy
python3 06_query_bandgap.py      # Classify by bandgap
python3 07_query_recent.py       # Recent calculations
python3 08_query_statistics.py   # Database statistics
```

### Using CLI

```bash
# Query by formula
atomate2siesta-database query Si

# List recent calculations
atomate2siesta-database list --limit 20

# Show database statistics
atomate2siesta-database stats
```

### Using Python (Maggma) - Basic Example

The tutorial scripts above show complete working examples. Here's the basic pattern:

```python
from maggma.stores import MongoStore

# Connect to database
store = MongoStore(
    database="atomate2siesta",
    collection_name="tasks",
    host="localhost",
    port=27017
)

store.connect()

# Count documents
count = store.count()
print(f"Total calculations: {count}")

# Get one document
doc = store.query_one()

if doc:
    # Handle nested document structure
    output = doc.get("output", {})
    if isinstance(output, dict):
        calc_output = output.get("output", {})
        formula = output.get("formula_pretty", "N/A")
        energy = calc_output.get("energy", "N/A")
        bandgap = calc_output.get("bandgap", "N/A")

        print(f"Formula: {formula}")
        print(f"Energy: {energy} eV")
        print(f"Bandgap: {bandgap} eV")

store.close()
```

### Query by Chemical Formula

```python
# Query by formula (see 04_query_by_formula.py)
formula = "Si"
docs = list(store.query(criteria={"output.formula_pretty": formula}))
print(f"Found {len(docs)} calculations for {formula}")

for doc in docs[:5]:
    output = doc.get("output", {})
    if isinstance(output, dict):
        calc_output = output.get("output", {})
        print(f"  Energy: {calc_output.get('energy', 'N/A')} eV")
        print(f"  Date: {doc.get('completed_at', 'N/A')}")
```

### Query by Energy Range

```python
# Query by energy range (see 05_query_by_energy.py)
min_energy = -230.0  # eV
max_energy = -220.0  # eV

docs = list(store.query(criteria={
    "output.output.energy": {"$gte": min_energy, "$lte": max_energy}
}))

# Sort by energy
def get_energy(doc):
    output = doc.get("output", {})
    if isinstance(output, dict):
        calc_output = output.get("output", {})
        return calc_output.get("energy", float("inf"))
    return float("inf")

docs_sorted = sorted(docs, key=get_energy)
```

### Classify by Bandgap

```python
# Classify materials (see 06_query_bandgap.py)
docs = list(store.query(criteria={"output.output.bandgap": {"$exists": True}}))

metals = []
semiconductors = []
insulators = []

for doc in docs:
    output = doc.get("output", {})
    if isinstance(output, dict):
        calc_output = output.get("output", {})
        bandgap = calc_output.get("bandgap")

        if bandgap is not None:
            if bandgap < 0.1:
                metals.append(doc)
            elif bandgap < 3.0:
                semiconductors.append(doc)
            else:
                insulators.append(doc)

print(f"Metals: {len(metals)}")
print(f"Semiconductors: {len(semiconductors)}")
print(f"Insulators: {len(insulators)}")
```

### Additional Query Patterns

```python
# Find calculations by directory
docs = store.query({"dir_name": {"$regex": "relax_"}})

# Find converged calculations
docs = store.query({"output.scf_converged": True})

# Find calculations with specific basis
docs = store.query({"input.parameters.PAO.BasisSize": "DZP"})

# Sort by date (most recent first)
docs_sorted = sorted(docs, key=lambda d: d.get("completed_at", ""), reverse=True)
```

---

## SiestaTaskDoc Schema

### Top-Level Fields

```python
{
    "task_id": "unique-uuid",
    "formula_pretty": "Si",
    "formula_anonymous": "A",
    "dir_name": "/path/to/calculation",
    "last_updated": "2025-10-22T12:00:00",
    "structure": {...},          # Final structure
    "input": {...},              # Input parameters
    "output": {...},             # Calculation output
    "calcs_reversed": [...]      # All calculation steps
}
```

### Output Fields

```python
doc["output"] = {
    "energy": -214.567,          # Final energy (eV)
    "energy_per_atom": -107.283,
    "scf_converged": True,
    "forces": [...],             # Atomic forces (eV/Ang)
    "forces_max": 0.001,
    "stress": [...],             # Stress tensor (GPa)
    "structure": {...}           # Final structure
}
```

### Input Fields

```python
doc["input"] = {
    "structure": {...},          # Initial structure
    "parameters": {
        "PAO.BasisSize": "DZP",
        "Mesh.Cutoff": "300 Ry",
        "kpts": [4, 4, 4],
        ...
    },
    "pseudopotentials": {...}
}
```

---

## Best Practices

### 1. Always Test Connection First

```bash
atomate2siesta-database test
```

### 2. Use Dry-Run Before Real Calculations

```python
# Validate workflow setup
maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
```

### 3. Organize with Custom Collection Names

```yaml
# Different projects in different collections
JOB_STORE:
  docs_store:
    collection_name: project_surfaces  # Surfaces project
    # collection_name: project_defects # Defects project
```

### 4. Regular Database Backups

```bash
# Backup MongoDB database
mongodump --db atomate2siesta --out /backup/path

# Restore from backup
mongorestore --db atomate2siesta /backup/path/atomate2siesta
```

### 5. Index Important Fields

```python
from pymongo import MongoClient, ASCENDING

client = MongoClient("localhost", 27017)
db = client["atomate2siesta"]
collection = db["tasks"]

# Create indexes for faster queries
collection.create_index([("formula_pretty", ASCENDING)])
collection.create_index([("last_updated", ASCENDING)])
collection.create_index([("dir_name", ASCENDING)])
```

---

## Troubleshooting

### Issue 1: Connection Failed

**Problem**: `atomate2siesta-database test` fails

**Solutions**:
```bash
# Check if MongoDB is running
mongosh --eval "db.version()"

# Start MongoDB
brew services start mongodb-community  # macOS
sudo systemctl start mongodb           # Linux

# Check port (default 27017)
netstat -an | grep 27017
```

### Issue 2: Configuration Not Found

**Problem**: No `~/.jobflow.yaml` file

**Solution**:
```bash
# Generate automatically
atomate2siesta-database config --generate

# Verify created
cat ~/.jobflow.yaml
```

### Issue 3: Results Not Stored

**Problem**: Calculations run but no database entries

**Solutions**:
1. Verify `~/.jobflow.yaml` exists and has correct format
2. Check MongoDB is running: `atomate2siesta-database test`
3. Use `run_locally()` not custom job execution
4. Check for error messages in calculation output

### Issue 4: Permission Denied

**Problem**: Cannot write to database

**Solutions**:
```bash
# Check MongoDB permissions
mongosh
> use atomate2siesta
> db.tasks.insertOne({test: 1})

# If fails, check MongoDB user permissions
# Or run MongoDB without authentication (local dev only):
# mongod --noauth
```

---

## Integration with HPC Clusters

### Local Development

Use local MongoDB for testing:

```yaml
# ~/.jobflow.yaml (local)
JOB_STORE:
  docs_store:
    host: localhost
    port: 27017
```

### Remote Cluster

Use cluster MongoDB or SSH tunnel:

```yaml
# ~/.jobflow.yaml (cluster)
JOB_STORE:
  docs_store:
    host: cluster-mongodb.university.edu
    port: 27017
    username: myuser
    password: mypass  # Or use environment variable
```

### SSH Tunnel

```bash
# Forward remote MongoDB to local port
ssh -L 27018:localhost:27017 user@cluster.edu

# Connect to forwarded port
atomate2siesta-database test --port 27018
```

---

## Advanced Usage

### Custom Query Filters

```python
# Complex queries with MongoDB operators
from maggma.stores import MongoStore

store = MongoStore(...)
store.connect()

# Multiple conditions (AND)
docs = store.query({
    "formula_pretty": "Si",
    "output.energy": {"$lt": -200},
    "input.parameters.PAO.BasisSize": "DZP"
})

# OR conditions
docs = store.query({
    "$or": [
        {"formula_pretty": "Si"},
        {"formula_pretty": "Ge"}
    ]
})

# Regex patterns
docs = store.query({
    "dir_name": {"$regex": ".*convergence.*"}
})
```

### Aggregation Pipeline

```python
# Group calculations by formula and count
pipeline = [
    {"$group": {
        "_id": "$formula_pretty",
        "count": {"$sum": 1},
        "avg_energy": {"$avg": "$output.energy_per_atom"}
    }},
    {"$sort": {"count": -1}}
]

results = list(collection.aggregate(pipeline))

for result in results:
    print(f"{result['_id']}: {result['count']} calculations, "
          f"avg E = {result['avg_energy']:.3f} eV/atom")
```

---

## Summary

### Key Takeaways

✅ **MongoDB** provides centralized storage for calculation results
✅ **atomate2siesta-database CLI** simplifies database management
✅ **~/.jobflow.yaml** configures database connection
✅ **SiestaTaskDoc** stores complete calculation data
✅ **maggma.stores** enables powerful Python queries

### Essential Workflow

```bash
# 1. Setup
atomate2siesta-database config --generate
atomate2siesta-database test

# 2. Run calculations (Python)
python my_workflow.py  # Uses ~/.jobflow.yaml automatically

# 3. Query results
atomate2siesta-database list
atomate2siesta-database query Si
atomate2siesta-database stats
```

### Next Steps

1. **Try the tutorial**: `python tutorial.py`
2. **Test different EXAMPLE_TYPEs**:
   - `'cli_tools'`: Explore CLI commands
   - `'basic_storage'`: Run calculation with database
   - `'query_results'`: Retrieve and analyze data
3. **See HPC integration**: `04-infrastructure/02-job-submission/`
4. **Build analysis workflows** using stored data

---

**Make database storage your default for all production calculations!** 🗄️
