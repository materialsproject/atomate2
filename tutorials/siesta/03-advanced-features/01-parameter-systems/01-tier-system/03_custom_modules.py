#!/usr/bin/env python
"""Fine-grained control over dataclass modules using enabled_modules.

This tutorial demonstrates how to customize which calculation parameters are active
by explicitly controlling module activation, giving you precise control beyond what
tier presets provide.

Purpose
-------
The tier system (basic/intermediate/advanced/expert) automatically activates different
sets of dataclass modules based on calculation complexity. Sometimes you need more
control - maybe you want a basic tier but with specific advanced features enabled,
or you want to disable certain modules to simplify input files.

The `enabled_modules` parameter lets you override the automatic tier-based module
selection and specify exactly which modules should be active.

What Are Modules?
-----------------
Modules are organized groups of FDF parameters implemented as dataclasses:
- scf_loop: SCF convergence parameters (MaxSCFIterations, DM.Tolerance, etc.)
- spin: Spin-polarization settings (Spin, DM.InitSpin, etc.)
- basis: Basis set parameters (PAO.BasisSize, PAO.EnergyShift, etc.)
- kpoints: k-point sampling (automatic k-grid generation)
- mesh: Real-space mesh parameters (Mesh.Cutoff, etc.)
- exchange_correlation: XC functional settings
- md_relaxation: Geometry optimization parameters (required for RelaxMaker)
- ... and 20+ more modules

When to Use This
----------------
1. **Minimal input files**: Enable only essential modules for cleaner FDF files
2. **Testing specific features**: Isolate parameters to test one aspect at a time
3. **Hybrid configurations**: Combine basic tier speed with specific advanced features
4. **Debugging**: Disable modules to identify parameter conflicts
5. **Custom workflows**: Build your own parameter "tier" from scratch

Example in This Tutorial
------------------------
We start with tier="basic" (normally activates 8 core modules) but override it
to enable ONLY the scf_loop module. This creates an extremely minimal calculation
with just SCF convergence parameters - useful for testing or learning what each
module contributes.

Available Modules (28 total)
-----------------------------
Core (tier: basic):
  - general_system, pseudopotentials, basis, kpoints, exchange_correlation,
    spin, scf_loop, mesh

Intermediate (tier: intermediate):
  - + hamiltonian_overlap, electronic_structure, density_of_states

Advanced (tier: advanced):
  - + chemical_analysis, optical_properties, grids, efficiency, external_control

Expert (tier: expert):
  - + auxiliary_forcefield, parallel, denchar, netcdf, constraints, phonon,
    dftu, rttddft, structural_v1, structural_v2, solvers

See Also
--------
- 04_tier_with_overrides.py: Override specific parameters within a tier
- atomate2siesta-presets list: View all available modules and their tier assignments
"""

from jobflow import run_locally
from pymatgen.core import Structure
from atomate2.siesta.jobs.core import RelaxMaker

structure = Structure.from_file("../../../00-structures/Si_mp-149_primitive.cif")

maker = RelaxMaker.fixed_cell_relaxation(
    tier="basic", enabled_modules=["scf_loop"], dry_run=True
)
job = maker.make(structure)
results = run_locally(job, create_folders=True)

print("✓ Basic tier + SCF control complete")
