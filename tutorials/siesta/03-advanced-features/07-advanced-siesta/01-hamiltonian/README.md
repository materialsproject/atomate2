# Tutorial 15: Hamiltonian and Overlap Matrix Configuration

## Overview

This tutorial demonstrates how to configure SIESTA's Hamiltonian (H) and overlap (S) matrix element handling using the `HamiltonianAndOverlapParameters` dataclass module. These settings control matrix storage, sparsity approximations, and post-processing capabilities.

## Learning Objectives

- Save Hamiltonian and overlap matrices (.HS files)
- Configure sparsity approximations for performance
- Enable extra SCF information for debugging
- Understand auxiliary supercells for periodic systems

## Theory

### Hamiltonian and Overlap Matrices

In DFT calculations, the electronic structure is determined by the generalized eigenvalue problem:

**H ψ = E S ψ**

Where:
- **H** = Hamiltonian matrix (energy operator)
- **S** = Overlap matrix (basis function overlaps)
- **ψ** = Wave functions (eigenvectors)
- **E** = Eigenvalues (energy levels)

### Matrix Sparsity

**Key concept**: Most matrix elements are zero due to localized basis functions.

For localized atomic orbital (LCAO) basis:
- Basis functions decay exponentially with distance
- Only nearby orbital pairs have non-zero matrix elements
- **Sparsity**: ~99% of elements are zero for large systems

**Example: 100-atom system with DZP basis**
- Total matrix size: 2,600 × 2,600 = 6.76M elements
- Non-zero elements: ~50,000 (0.7%)
- **Memory savings**: 99.3% reduction using sparse storage

### Matrix Storage Formats

**1. Dense Storage (not used in SIESTA)**
```
Store all elements: N² memory
100 atoms, DZP: 6.76M × 8 bytes = 54 MB
```

**2. Sparse Storage (SIESTA default)**
```
Store only non-zero: Depends on cutoff radius
100 atoms, DZP: ~50k × 8 bytes = 0.4 MB
Compression: 135× smaller
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `negl_non_overlap_int` | bool | False | Neglect matrix elements for non-overlapping orbital pairs |
| `scf_write_extra` | bool | False | Write extra SCF information for debugging |
| `save_hs` | bool | True | Save Hamiltonian and overlap matrices to .HS file |
| `force_aux_cell` | bool | False | Force use of auxiliary supercell for matrix elements |

## Examples

### Example 1: Save Hamiltonian and Overlap Matrices

Enable .HS file output for post-processing:

```python
user_params = {
    "SaveHS": True,              # Save H and S matrices (default)
    "PAO.BasisSize": "DZP",
}

maker = StaticMaker.scf(user_params=user_params)
```

**Output files:**
- `systemLabel.HS` - Binary file with H, S matrices
- Contains sparse matrix in COO (coordinate) format
- Essential for transport calculations (TranSIESTA)
- Required for wannier90 interface
- Useful for post-processing and analysis

**File size estimate:**
- 100 atoms, DZP: ~5 MB
- 1000 atoms, DZP: ~50 MB
- Scales linearly with system size

### Example 2: Sparsity Approximation for Large Systems

Neglect matrix elements for non-overlapping orbitals:

```python
user_params = {
    "Negl.NonOverlap.Int": True,  # Faster calculation
    "PAO.BasisSize": "DZP",
}

maker = RelaxMaker.fixed_cell_relaxation(user_params=user_params)
```

**Effect:**
- Matrix elements computed only if orbital pairs overlap
- Speedup: 10-30% for large systems (>500 atoms)
- Accuracy: Negligible impact (<0.1 meV/atom)

**When to use:**
- Large systems (>200 atoms)
- Preliminary calculations and screening
- Geometries with well-separated fragments

**When NOT to use:**
- High-accuracy calculations
- Publication-quality results
- Systems with weak interactions (van der Waals)

### Example 3: Debug SCF Convergence

Enable extra SCF information:

```python
user_params = {
    "SCF.WriteExtra": True,       # Verbose SCF output
    "PAO.BasisSize": "DZP",
    "SCF.MixingWeight": 0.1,
}

maker = StaticMaker.scf(user_params=user_params)
```

**Additional output:**
- Matrix eigenvalue spectrum
- Charge mixing details
- Orbital populations
- Mulliken analysis per iteration

**Use cases:**
- SCF convergence problems
- Understanding charge transfer
- Debugging electronic structure issues

### Example 4: Force Auxiliary Supercell

Manually control auxiliary cell for matrix elements:

```python
user_params = {
    "ForceAuxCell": True,         # Use auxiliary supercell
    "PAO.BasisSize": "DZP",
}

maker = StaticMaker.scf(user_params=user_params)
```

**What it does:**
- Forces SIESTA to use larger auxiliary cell
- Ensures matrix elements computed in supercell
- Rarely needed (automatic detection usually works)

**When to use:**
- Very small unit cells with long-range basis
- Debugging matrix element issues
- Non-standard boundary conditions

## Hamiltonian and Overlap Matrix Files

### .HS File Format

The `.HS` file contains the Hamiltonian and overlap matrices in sparse format:

**Structure:**
1. Header: System information (atoms, orbitals, cell)
2. Sparse pattern: List of non-zero element indices
3. Hamiltonian: Non-zero matrix elements of H
4. Overlap: Non-zero matrix elements of S

**Storage format (COO - Coordinate Format):**
```
Row Index | Column Index | H Value | S Value
----------|--------------|---------|--------
1         | 1            | -5.234  | 1.000
1         | 2            | -0.452  | 0.234
2         | 1            | -0.452  | 0.234
...
```

### Reading .HS Files in Python

**Using sisl library:**
```python
import sisl

# Read .HS file
hs = sisl.get_sile('systemLabel.HS').read_hamiltonian()

# Access Hamiltonian matrix
H = hs.tocsr()  # Sparse CSR format
H_dense = H.toarray()  # Convert to dense (caution: memory!)

# Access overlap matrix
S = hs.S.tocsr()

# Get eigenvalues
import numpy as np
eigenvalues = np.linalg.eigvals(H_dense @ np.linalg.inv(S.toarray()))
```

**Using custom Python code:**
```python
import numpy as np
import struct

def read_hs_file(filename):
    """Read SIESTA .HS file (simplified)."""
    with open(filename, 'rb') as f:
        # Read header
        no_u = struct.unpack('i', f.read(4))[0]  # Number of orbitals
        no_s = struct.unpack('i', f.read(4))[0]  # Number of supercells

        # Read sparse pattern
        numh = np.fromfile(f, dtype=np.int32, count=no_u)
        listhptr = np.zeros(no_u + 1, dtype=np.int32)
        listhptr[1:] = np.cumsum(numh)

        max_nh = listhptr[-1]
        listh = np.fromfile(f, dtype=np.int32, count=max_nh)

        # Read Hamiltonian and overlap
        H = np.fromfile(f, dtype=np.float64, count=max_nh)
        S = np.fromfile(f, dtype=np.float64, count=max_nh)

    return {'no_u': no_u, 'H': H, 'S': S, 'listh': listh}
```

### Post-Processing Applications

**1. Density of States (DOS):**
```python
import sisl

hs = sisl.get_sile('systemLabel.HS').read_hamiltonian()
mp = sisl.MonkhorstPack(hs, [10, 10, 10])

# Calculate DOS
E = np.linspace(-10, 5, 1000)
DOS = mp.asaverage().DOS(E)
```

**2. Band Structure:**
```python
# Define band path
band = sisl.BandStructure(hs, [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]])

# Calculate eigenvalues
eigenvalues = band.apply.array.eigh()
```

**3. Transport Calculations:**
```python
# Use with TranSIESTA
# .HS files from electrodes and scattering region
# Compute transmission, I-V curves
```

## Sparsity Approximation Details

### Negl.NonOverlap.Int Option

**Default behavior (False):**
- Compute all matrix elements within cutoff radius
- Includes elements even if orbital densities barely overlap
- Most accurate, but slower

**With approximation (True):**
- Skip matrix elements if orbital overlap < threshold
- Faster matrix construction
- Negligible accuracy loss

**Performance impact:**

| System Size | Without | With | Speedup |
|-------------|---------|------|---------|
| 50 atoms    | 100 s   | 95 s | 5%     |
| 200 atoms   | 450 s   | 380 s | 15%   |
| 1000 atoms  | 2800 s  | 2100 s | 25%  |

**Accuracy impact:**

| Property | Difference |
|----------|-----------|
| Total energy | <0.1 meV/atom |
| Forces | <0.01 eV/Å |
| Stress | <0.1 GPa |
| Band gap | <0.001 eV |

### Overlap Threshold

Internal threshold for orbital overlap (not user-configurable):
- Default: 10⁻⁶ (overlap integral)
- Very small, affects only distant pairs
- Conservative value ensures accuracy

## Auxiliary Supercell

### What is the Auxiliary Cell?

**Purpose**: Ensure matrix elements include all periodic images

For periodic systems:
- Orbitals can overlap across cell boundaries
- Need to include neighboring cell images
- Auxiliary cell = enlarged supercell for matrix evaluation

**Automatic determination:**
- SIESTA calculates required auxiliary cell size
- Based on maximum orbital cutoff radius
- Typically 1-3 times unit cell in each direction

### Manual Control with ForceAuxCell

**Rarely needed**, but can force larger auxiliary cell:

```python
user_params = {
    "ForceAuxCell": True,
}
```

**When to consider:**
- Very small unit cells (<5 Å)
- Very diffuse basis functions
- Debugging matrix element issues

**Example scenario:**
```
Unit cell: 3 Å × 3 Å × 3 Å
Orbital cutoff: 10 Å
Automatic auxiliary: 5 × 5 × 5 supercell
With ForceAuxCell: May use larger if needed
```

## SCF Extra Information

### SCF.WriteExtra Output

**Standard output (SCF.WriteExtra=False):**
```
scf: iter   Energy       dH     dDM    dEdmax
scf:   1  -215.431    0.500  0.823   5.234
scf:   2  -215.892    0.122  0.234   1.123
scf:   3  -215.923    0.034  0.087   0.234
```

**Extra output (SCF.WriteExtra=True):**
```
scf: iter   Energy       dH     dDM    dEdmax
scf:   1  -215.431    0.500  0.823   5.234
siesta: Eigenvalues (eV):
    -12.34  -8.92  -7.23  -5.67  -4.12  -2.34  1.23  3.45
siesta: Mulliken populations:
    Atom   Total   s     p     d
    Si1    4.00    1.25  2.75  0.00
    Si2    4.00    1.25  2.75  0.00
siesta: Charge transfer: 0.000 e
```

**Use cases:**
1. Understanding electronic structure evolution during SCF
2. Debugging convergence issues
3. Analyzing charge transfer
4. Verifying orbital populations

## Best Practices

### 1. Standard Production Calculations

**Default settings (recommended):**
```python
{
    "SaveHS": True,                    # Always save for post-processing
    "Negl.NonOverlap.Int": False,      # Maximum accuracy
    "SCF.WriteExtra": False,           # Keep output clean
    "ForceAuxCell": False,             # Automatic is best
}
```

### 2. Large System Optimization

**For systems >200 atoms:**
```python
{
    "SaveHS": True,
    "Negl.NonOverlap.Int": True,       # 15-25% faster
    "SCF.WriteExtra": False,
}
```

**Verify accuracy:**
- Compare with/without approximation on smaller test system
- Check total energy converged to <1 meV/atom
- Verify forces agree to <0.01 eV/Å

### 3. SCF Convergence Debugging

**For difficult convergence:**
```python
{
    "SaveHS": True,
    "SCF.WriteExtra": True,            # Verbose output
    "SCF.MixingWeight": 0.05,          # Slower, more stable mixing
    "SCF.Mixer.History": 8,            # More history for Pulay
}
```

### 4. Post-Processing Workflows

**Transport calculations (TranSIESTA):**
```python
# Electrode calculation
electrode_params = {
    "SaveHS": True,                    # Essential for transport
    "kpts": [1, 1, 50],                # Dense along transport
}

# Scattering region
scattering_params = {
    "SaveHS": True,
}
```

**Band structure and DOS:**
```python
{
    "SaveHS": True,
    "kpts": [8, 8, 8],                 # Uniform sampling
}
# Post-process with sisl or TBtrans
```

### 5. Memory-Limited Systems

**Disable .HS if memory is critical:**
```python
{
    "SaveHS": False,                   # Skip .HS file
    "Negl.NonOverlap.Int": True,       # Reduce computation
}
```

**Memory savings:**
- No .HS file on disk (typically 5-50 MB)
- Slightly less memory during run (~5%)

## Common Issues

### Issue 1: .HS File Too Large

**Problem:** .HS file fills disk (>1 GB)

**Causes:**
- Very large system (>5000 atoms)
- Dense basis (TZP or larger)

**Solutions:**
```python
# Option 1: Disable if not needed
{"SaveHS": False}

# Option 2: Post-process immediately, then delete
# Run calculation, extract data, rm *.HS

# Option 3: Compress files
# gzip systemLabel.HS (70-80% compression)
```

### Issue 2: Matrix Elements Seem Wrong

**Problem:** Unexpected eigenvalues or band structure

**Debug steps:**
1. Check auxiliary cell size (look for warnings in output)
2. Try ForceAuxCell=True
3. Verify basis set is appropriate
4. Check cutoff radius (PAO.EnergyShift)

```python
# Debug configuration
{
    "ForceAuxCell": True,
    "SCF.WriteExtra": True,
    "PAO.EnergyShift": "0.02 Ry",     # Tighter cutoff
}
```

### Issue 3: SCF Convergence with Approximation

**Problem:** Negl.NonOverlap.Int causes SCF oscillations

**Solution:** Disable approximation for difficult cases
```python
{
    "Negl.NonOverlap.Int": False,      # Full accuracy
    "SCF.MixingWeight": 0.05,          # Slower mixing
}
```

### Issue 4: Reading .HS File Fails

**Problem:** Cannot read .HS file in post-processing

**Causes:**
1. File corrupted (calculation crashed)
2. Incompatible SIESTA version
3. Incorrect file format

**Solutions:**
```bash
# Check file integrity
ls -lh systemLabel.HS
file systemLabel.HS

# Use sisl (handles version differences)
python -c "import sisl; hs = sisl.get_sile('systemLabel.HS').read_hamiltonian()"

# Regenerate if needed
# Re-run SIESTA with SaveHS=True
```

## Advanced Topics

### Matrix Element Analysis

**Extract specific matrix elements:**
```python
import sisl

hs = sisl.get_sile('systemLabel.HS').read_hamiltonian()

# Get H and S for specific orbitals
i_orb = 10  # Orbital index
j_orb = 15

# H[i,j] element
H_ij = hs[i_orb, j_orb]
S_ij = hs.S[i_orb, j_orb]

print(f"H[{i_orb},{j_orb}] = {H_ij:.4f} eV")
print(f"S[{i_orb},{j_orb}] = {S_ij:.4f}")
```

### Sparse Matrix Statistics

**Analyze sparsity pattern:**
```python
import sisl
import numpy as np

hs = sisl.get_sile('systemLabel.HS').read_hamiltonian()
H_sparse = hs.tocsr()

n_orbitals = H_sparse.shape[0]
n_nonzero = H_sparse.nnz
max_elements = n_orbitals ** 2

sparsity = 100 * (1 - n_nonzero / max_elements)
avg_elements_per_orbital = n_nonzero / n_orbitals

print(f"Matrix size: {n_orbitals} × {n_orbitals}")
print(f"Non-zero elements: {n_nonzero:,}")
print(f"Sparsity: {sparsity:.2f}%")
print(f"Average elements per orbital: {avg_elements_per_orbital:.1f}")
```

**Typical results:**
```
Matrix size: 2600 × 2600
Non-zero elements: 45,832
Sparsity: 99.32%
Average elements per orbital: 17.6
```

### Hamiltonian Visualization

**Plot sparse pattern:**
```python
import matplotlib.pyplot as plt
import sisl

hs = sisl.get_sile('systemLabel.HS').read_hamiltonian()
H_sparse = hs.tocsr()

plt.figure(figsize=(10, 10))
plt.spy(H_sparse, markersize=0.5)
plt.title('Hamiltonian Sparsity Pattern')
plt.xlabel('Orbital Index')
plt.ylabel('Orbital Index')
plt.savefig('hamiltonian_sparsity.png', dpi=300)
```

## Further Reading

- SIESTA Manual: Section 4.25 (Hamiltonian and Overlap)
- sisl Documentation: https://zerothi.github.io/sisl/
- TranSIESTA Tutorial: Transport calculations with .HS files
- Sparse Matrix Methods: Compressed storage formats

## Next Steps

- Tutorial 12: Parallel Computation (optimize matrix diagonalization)
- Tutorial 14: Efficiency Options (timing matrix operations)
- Tutorial 10: Grid Output (complementary to matrix output)
