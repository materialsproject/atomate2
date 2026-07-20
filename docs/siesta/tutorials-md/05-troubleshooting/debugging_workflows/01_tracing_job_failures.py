"""
Debugging Workflows - Tracing Job Failures
==========================================

This tutorial demonstrates how to debug workflow failures systematically:
- Identifying which job failed in a flow
- Examining job outputs and intermediate results
- Using dry-run mode to test workflows
- Database queries for job history (jobflow-remote)

Category: troubleshooting
Difficulty: Intermediate
Time: 20 minutes
"""


from jobflow import Flow, run_locally
from pymatgen.core import Lattice, Structure

from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker

# =============================================================================
# Step 1: Understanding Workflow Structure
# =============================================================================

print("=" * 70)
print("Step 1: Understanding Workflow Structure")
print("=" * 70)

# Create a simple two-step workflow
si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])

# Create makers
relax_maker = RelaxMaker.fixed_cell_relaxation(dry_run=True)
static_maker = StaticMaker(dry_run=True)

# Create jobs
relax_job = relax_maker.make(si)
static_job = static_maker.make(si)

# Create flow with dependencies
flow = Flow([relax_job, static_job], name="Debug Example Flow")

print(f"Flow name: {flow.name}")
print(f"Number of jobs: {len(flow.jobs)}")
for i, job in enumerate(flow.jobs):
    print(f"  Job {i + 1}: {job.name} (UUID: {job.uuid[:8]}...)")


# =============================================================================
# Step 2: Running with Verbose Output
# =============================================================================

print("\n" + "=" * 70)
print("Step 2: Running with Verbose Output")
print("=" * 70)

# Run locally with folder creation for inspection
responses = run_locally(flow, create_folders=True, raise_immediately=False)

print("\nJob responses:")
for uuid, response_list in responses.items():
    # Get job name
    job_name = (
        [j.name for j in flow.jobs if j.uuid == uuid][0] if flow.jobs else "Unknown"
    )

    # Handle response - can be Response object or int
    if isinstance(response_list, list):
        for response in response_list:
            if hasattr(response, "output"):
                print(f"  {job_name}: Success (has output)")
            else:
                print(f"  {job_name}: {response}")
    else:
        print(f"  {job_name}: {response_list}")


# =============================================================================
# Step 3: Examining Job Directories
# =============================================================================

print("\n" + "=" * 70)
print("Step 3: Examining Job Directories")
print("=" * 70)

debugging_tips = """
After running a workflow locally, check the job directories:

# List job directories
ls -ltr job_*/

# Check which job directory corresponds to which job
# Each directory is named with the job UUID

# Examine SIESTA output for errors
grep -i "error\\|fail\\|abort" job_*/siesta.out

# Check input parameters
cat job_*/siesta.fdf | head -50

# Check SCF convergence
grep "scf:" job_*/siesta.out | tail -20

# Check geometry optimization
grep "siesta: E_KS(eV)" job_*/siesta.out
"""
print(debugging_tips)


# =============================================================================
# Step 4: Using Dry-Run Mode for Testing
# =============================================================================

print("\n" + "=" * 70)
print("Step 4: Using Dry-Run Mode for Testing")
print("=" * 70)

print(
    """
Dry-run mode is essential for debugging workflows:

1. Generates complete input files without running SIESTA
2. Validates parameters before expensive calculations
3. Allows checking workflow structure and dependencies

Usage:
"""
)

# Example: Test a problematic workflow
test_maker = RelaxMaker.fixed_cell_relaxation(
    dry_run=True,
    user_params={
        "PAO.BasisSize": "TZP",
        "a2s_kpts": [6, 6, 6],
        "Mesh.Cutoff": "300 Ry",
    },
)

test_job = test_maker.make(si)
print(f"Created test job: {test_job.name}")
print("Running in dry-run mode to verify parameters...")

test_response = run_locally(test_job, create_folders=True)
print("Dry-run completed! Check generated files.")


# =============================================================================
# Step 5: Jobflow-Remote Debugging
# =============================================================================

print("\n" + "=" * 70)
print("Step 5: Jobflow-Remote Debugging (HPC)")
print("=" * 70)

jobflow_remote_tips = """
For jobflow-remote (HPC) debugging:

# List failed jobs
jf -p PROJECT job list --state FAILED

# Get detailed job information
jf -p PROJECT job info <db_id> --full

# Check job error output
jf -p PROJECT job output <db_id>

# Re-run a failed job with different parameters
jf -p PROJECT job rerun <db_id>

# Database queries for job history
from jobflow_remote import JobManager

jm = JobManager(project="PROJECT")

# Find all failed jobs for a material
failed = jm.query_jobs(
    {"output.formula_pretty": "Si", "state": "FAILED"}
)

# Get job details
for job in failed:
    print(f"Job: {job['name']}, Error: {job.get('error', 'Unknown')}")

# Check intermediate outputs
completed = jm.query_jobs(
    {"output.formula_pretty": "Si", "state": "COMPLETED"}
)
for job in completed:
    output = job.get("output", {})
    energy = output.get("energy")
    print(f"Job: {job['name']}, Energy: {energy} eV")
"""
print(jobflow_remote_tips)


# =============================================================================
# Step 6: Common Debugging Patterns
# =============================================================================

print("\n" + "=" * 70)
print("Step 6: Common Debugging Patterns")
print("=" * 70)

patterns = """
Pattern 1: Job failed but no error message
------------------------------------------
- Check if SIESTA actually ran: ls job_*/siesta.out
- Check stderr: cat job_*/stderr.txt (if exists)
- Check resource limits: Was job killed by scheduler?

Pattern 2: SCF oscillating
--------------------------
grep "scf:" job_*/siesta.out
# Look for energy going up and down
# Fix: Reduce SCF.Mixer.Weight to 0.01-0.05

Pattern 3: Memory error
-----------------------
# Check for OOM killer
dmesg | grep -i "killed process"
# Or scheduler killed it
grep -i "memory\\|killed" job_*/slurm*.out

Pattern 4: Wrong structure after relaxation
-------------------------------------------
from pymatgen.io.siesta import SiestaOutput
output = SiestaOutput("job_*/siesta.out")
final_structure = output.final_structure
# Compare with initial
initial_structure.get_distance(0, 1)  # Bond lengths
final_structure.get_distance(0, 1)

Pattern 5: Workflow dependency issue
------------------------------------
# Job 2 didn't receive output from Job 1
# Check if Job 1 completed successfully
# Check if output reference is correct in flow definition
"""
print(patterns)


# =============================================================================
# Step 7: Debugging Checklist
# =============================================================================

print("\n" + "=" * 70)
print("Step 7: Debugging Checklist")
print("=" * 70)

checklist = """
DEBUGGING CHECKLIST
===================

Before Running:
[ ] Test with dry_run=True first
[ ] Verify structure is reasonable (no overlapping atoms)
[ ] Check k-points are appropriate for system size
[ ] Ensure pseudopotentials are installed

After Failure:
[ ] Identify which job failed (UUID in error message)
[ ] Read error message in siesta.out (tail -100 job_*/siesta.out)
[ ] Check if SCF converged (grep "scf:" job_*/siesta.out)
[ ] Verify input file parameters (cat job_*/siesta.fdf)
[ ] Check resource usage (memory, time)
[ ] Try with simplified parameters (SZ basis, fewer k-points)

For HPC (jobflow-remote):
[ ] Check job state: jf -p PROJECT job list --state FAILED
[ ] Get full output: jf -p PROJECT job info <db_id> --full
[ ] Check cluster logs: Check slurm*.out or PBS logs
[ ] Verify file transfer: Were input files copied correctly?

If Still Stuck:
[ ] Create minimal reproducible example
[ ] Check SIESTA mailing list for similar issues
[ ] Open GitHub issue with:
    - Error message
    - siesta.fdf (input file)
    - Structure file
    - atomate2siesta version
"""
print(checklist)


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print(
    """
Key debugging strategies:

1. ALWAYS use dry_run=True first to validate parameters
2. Examine job directories for output and error files
3. Use grep to find errors and convergence issues
4. For HPC: Use jf commands to query job status
5. Follow systematic debugging checklist
6. One change at a time when fixing issues
7. Document what works for similar systems

Next tutorials:
- 02_analyzing_intermediate_outputs.py - Deep dive into outputs
- ../performance_optimization/ - Making calculations faster
"""
)
