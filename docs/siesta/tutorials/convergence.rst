==========================
Convergence Learning Paths
==========================

Systematic parameter optimization for production-quality calculations. These tutorials are
**essential** before running any production workflows.

----

Why Convergence Testing Matters
================================

**The Problem**:

DFT calculations depend on numerical approximations with adjustable parameters. Without
proper convergence testing, results may be:

* Inaccurate (errors > 10 meV/atom)
* Non-reproducible (different parameters → different results)
* Unreliable for comparisons

**The Solution**:

Systematically test parameters until results are converged within acceptable tolerances.

**Recommended Workflow**:

1. Run convergence studies once for your material class
2. Apply converged parameters to all production calculations
3. Document parameters in your workflow scripts

----

Tutorial 06: K-points and Mesh Cutoff
======================================

**Learning Objectives**:

* Understand k-point sampling and real-space grid cutoff
* Perform systematic convergence studies
* Determine production parameters
* Balance accuracy vs. computational cost

**Key Concepts**:

* **K-points**: Brillouin zone sampling for periodic systems
* **Mesh cutoff**: Real-space grid fineness for charge density
* Convergence criterion: Energy change < threshold

**Recommended Convergence Criteria**:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Property
     - Criterion
     - Typical Value
   * - Total energy
     - < 1 meV/atom
     - Δ kpts < 0.001 eV/atom
   * - Forces
     - < 0.01 eV/Å
     - For relaxations
   * - Band gap
     - < 0.01 eV
     - For semiconductors
   * - Lattice parameters
     - < 0.01 Å
     - For structure prediction

**K-point Convergence Example**:

.. code-block:: python

   from atomate2.siesta.flows.convergence import KpointsConvergenceFlowMaker
   from jobflow import run_locally

   # Test k-point meshes
   kpts_list = [
       [2, 2, 2],
       [4, 4, 4],
       [6, 6, 6],
       [8, 8, 8],
       [10, 10, 10],
   ]

   maker = KpointsConvergenceFlowMaker(
       kpoints_list=kpts_list,
   )

   flow = maker.make(structure)
   results = run_locally(flow, create_folders=True)

**Mesh Cutoff Convergence Example**:

.. code-block:: python

   from atomate2.siesta.flows.convergence import MeshCutoffConvergenceFlowMaker
   from jobflow import run_locally

   # Test cutoff values (Ry)
   mesh_cutoffs = [200, 250, 300, 350, 400, 450, 500]

   maker = MeshCutoffConvergenceFlowMaker(
       mesh_cutoffs=mesh_cutoffs,
   )

   flow = maker.make(structure)
   results = run_locally(flow, create_folders=True)

**Combined Mesh Cutoff + K-points Convergence**:

For efficient two-stage convergence testing with intelligent stopping:

.. code-block:: python

   from atomate2.siesta.flows.convergence import (
       MeshKpointConvergenceFlowMaker,
       ConvergenceCriteria,
   )
   from jobflow import run_locally

   # Define convergence criteria (multi-property testing)
   criteria = ConvergenceCriteria(
       energy_tol=1.0,      # 1 meV energy difference
       fermi_tol=0.01,      # 0.01 eV Fermi energy difference (optional)
       force_tol=0.01,      # 0.01 eV/Å maximum force (optional)
       stress_tol=0.05,     # 0.05 eV/Å³ maximum stress (optional)
       bandgap_tol=0.01,    # 0.01 eV band gap difference (optional)
   )

   # Create combined workflow
   maker = MeshKpointConvergenceFlowMaker(
       mesh_cutoffs=[200, 250, 300, 350, 400, 450, 500],
       kpoints_list=[[2,2,2], [4,4,4], [6,6,6], [8,8,8], [10,10,10]],
       stage1_kpoints=[4, 4, 4],  # Coarse k-points for mesh convergence
       convergence_criteria=criteria,
       require_consecutive=2,      # Stop after 2 consecutive converged points
   )

   flow = maker.make(structure)
   results = run_locally(flow, create_folders=True)

   # Workflow stops testing when ALL criteria are met for consecutive points
   # Saves 30-50% of unnecessary calculations!

**How the Combined Workflow Works**:

1. **Stage 1**: Tests mesh cutoff values sequentially with coarse k-points (4×4×4)
2. **Stage 2**: Tests k-point grids sequentially using converged mesh cutoff from Stage 1
3. **Intelligent Stopping**: Stops testing when ALL specified criteria are satisfied for
   two consecutive parameter values (configurable via ``require_consecutive``)
4. **Comprehensive Output**:

   * 6 individual plots per stage: energy, convergence, Fermi energy, bandgap, forces, stress
   * Detailed text files with convergence analysis
   * Automatic parameter evolution tracking

**Output Files Generated** (per stage):

* ``convergence_mesh_cutoff_energy.png`` - Total energy vs. parameter
* ``convergence_mesh_cutoff_convergence.png`` - Energy difference between consecutive points
* ``convergence_mesh_cutoff_fermi.png`` - Fermi energy evolution
* ``convergence_mesh_cutoff_bandgap.png`` - Band gap evolution (if applicable)
* ``convergence_mesh_cutoff_forces.png`` - Maximum force evolution
* ``convergence_mesh_cutoff_stress.png`` - Maximum stress evolution
* ``convergence_mesh_cutoff.txt`` - Comprehensive text analysis

Same pattern for k-points stage: ``convergence_kpoints_*.png/txt``

**Use Case Examples**:

1. **Energy-Only Convergence** (fastest):

   .. code-block:: python

      # Default criteria - only energy testing
      maker = MeshKpointConvergenceFlowMaker(
          mesh_cutoffs=[200, 300, 400, 500],
          kpoints_list=[[4,4,4], [6,6,6], [8,8,8]],
      )
      # Uses default: ConvergenceCriteria(energy_tol=1.0)

2. **Relaxation-Focused Convergence** (tight forces):

   .. code-block:: python

      relax_criteria = ConvergenceCriteria(
          energy_tol=5.0,      # Looser energy (5 meV)
          force_tol=0.01,      # Tight forces (critical for relaxations)
          stress_tol=0.05,     # For variable-cell relaxations
      )

      maker = MeshKpointConvergenceFlowMaker(
          mesh_cutoffs=[200, 300, 400, 500],
          kpoints_list=[[4,4,4], [6,6,6], [8,8,8]],
          convergence_criteria=relax_criteria,
      )

3. **Electronic Structure Convergence** (tight Fermi/bandgap):

   .. code-block:: python

      electronic_criteria = ConvergenceCriteria(
          energy_tol=1.0,
          fermi_tol=0.005,     # Very tight Fermi energy (for band structures)
          bandgap_tol=0.01,    # Tight band gap (for semiconductors)
      )

      maker = MeshKpointConvergenceFlowMaker(
          mesh_cutoffs=[200, 300, 400, 500],
          kpoints_list=[[4,4,4], [6,6,6], [8,8,8]],
          convergence_criteria=electronic_criteria,
      )

4. **Multi-Criteria Convergence** (comprehensive):

   .. code-block:: python

      strict_criteria = ConvergenceCriteria(
          energy_tol=1.0,      # All properties must be converged
          fermi_tol=0.01,
          force_tol=0.01,
          stress_tol=0.05,
          bandgap_tol=0.01,
      )

      maker = MeshKpointConvergenceFlowMaker(
          mesh_cutoffs=[150, 200, 250, 300, 350, 400, 450],
          kpoints_list=[[2,2,2], [3,3,3], [4,4,4], [6,6,6], [8,8,8], [10,10,10]],
          convergence_criteria=strict_criteria,
          require_consecutive=2,  # Require 2 consecutive converged points
      )

**Tutorials**:

See ``tutorials/02-convergence/01-kpoints-mesh-cutoff/`` for complete examples:

* ``04_1_combined_basic.py`` - Basic energy-only convergence
* ``04_2_combined_multi_criteria.py`` - Multi-property convergence
* ``04_3_combined_relaxation.py`` - Optimized for geometry relaxations
* ``04_4_combined_electronic.py`` - Optimized for band structure/DOS

**Guidelines**:

* **Metals**: Require denser k-points than insulators (Fermi surface)
* **2D materials**: Dense in-plane (xy), sparse out-of-plane (z)
* **Molecules**: Γ-point only (no periodicity)

**Typical Converged Values**:

* **K-points**: 4×4×4 (simple), 6×6×6 (standard), 8×8×8+ (high accuracy)
* **Mesh cutoff**: 300 Ry (standard), 400-500 Ry (high accuracy)

📁 **Location**: ``tutorials/02-convergence/01-kpoints-cutoff/``

⏱️ **Time**: 45 minutes

⭐ **Difficulty**: Intermediate

----

Tutorial 07: Basis Parameters (PAO.EnergyShift, SplitNorm)
===========================================================

**Learning Objectives**:

* Understand SIESTA's PAO basis set construction
* Converge PAO.EnergyShift and PAO.SplitNorm
* Balance basis quality vs. computational cost
* Analyze timing data

**Key Concepts**:

* **PAO (Pseudo-Atomic Orbital)**: SIESTA's numerical atomic orbitals
* **PAO.EnergyShift**: Orbital confinement energy (smaller = larger basis)
* **PAO.SplitNorm**: Multiple-zeta splitting parameter
* **Timing analysis**: Computational cost vs. accuracy tradeoff

**PAO.EnergyShift**:

Controls orbital confinement radius. Smaller values → larger orbitals → better accuracy but higher cost.

* Default: 0.02 Ry (~270 meV)
* Standard: 0.01 Ry (~136 meV)
* High accuracy: 0.005 Ry (~68 meV)
* Ultra-high: 0.001 Ry (~14 meV)

**PAO.SplitNorm**:

Controls splitting of orbitals for multiple-zeta basis sets (DZ, TZ).

* Default: 0.15
* Standard: 0.15-0.25
* Larger values → more localized split orbitals

**Example**:

.. code-block:: python

   from atomate2.siesta.flows.basis import BasisParametersConvergenceMaker

   # Test PAO.EnergyShift and PAO.SplitNorm together in a grid
   maker = BasisParametersConvergenceMaker(
       energy_shifts=[0.01, 0.015, 0.02],     # PAO.EnergyShift values
       split_norms=[0.15, 0.20, 0.25],        # PAO.SplitNorm values
       basis_size="DZP",                      # Fixed basis size
       kpts=[4, 4, 4],                        # K-points
   )

   flow = maker.make(structure)
   # This creates a 3×3 = 9 calculations testing all combinations

**Timing Analysis**:

Tutorial includes automatic timing analysis:

* Wall time vs. parameter value
* Scaling behavior
* Cost vs. accuracy tradeoff
* Optimal parameter selection

**Guidelines**:

1. Start with PAO.BasisSize (SZ → DZ → DZP → TZP)
2. Then converge PAO.EnergyShift for chosen basis size
3. Finally test PAO.SplitNorm for multiple-zeta (DZ, TZ)

**Typical Converged Values**:

* **Quick tests**: PAO.EnergyShift = 0.02 Ry
* **Standard**: PAO.EnergyShift = 0.01 Ry, SplitNorm = 0.15
* **High accuracy**: PAO.EnergyShift = 0.005 Ry, SplitNorm = 0.20

📁 **Location**: ``tutorials/02-convergence/02-basis-parameters/``

⏱️ **Time**: 60 minutes

⭐ **Difficulty**: Advanced

----

Tutorial 08: Complete Basis Convergence
========================================

**Learning Objectives**:

* Perform comprehensive basis set convergence
* Test basis size, energy shift, and split norm systematically
* Understand parameter interactions
* Select optimal basis parameters for production

**Workflow**:

1. **Basis Size**: SZ → DZ → DZP → TZP → QZP
2. **Energy Shift** (for each size): 0.05 → 0.02 → 0.01 → 0.005 Ry
3. **Split Norm** (for multi-zeta): 0.10 → 0.15 → 0.20 → 0.25

**Example**:

.. code-block:: python

   from atomate2.siesta.flows.basis import CompleteBasisConvergenceMaker

   maker = CompleteBasisConvergenceMaker(
       basis_sizes=["DZ", "DZP", "TZP"],
       energy_shifts=[0.01, 0.015, 0.02],
       split_norms=[0.15, 0.20, 0.25],
       kpts=[4, 4, 4],
   )

   flow = maker.make(structure)
   # This creates 3 basis × 3 shifts × 3 norms = 27 calculations

**Parameter Interactions**:

* Basis size has largest effect (SZ vs. DZP: ~10-100 meV/atom)
* EnergyShift has moderate effect (0.02 vs. 0.005 Ry: ~1-10 meV/atom)
* SplitNorm has small effect (~0.1-1 meV/atom)

**Basis Size Guidelines**:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Basis Size
     - Description
     - Use Case
   * - SZ
     - Single-zeta
     - Quick tests only (not production)
   * - DZ
     - Double-zeta
     - Minimum for production
   * - DZP
     - DZ + polarization
     - Standard production (recommended)
   * - TZP
     - Triple-zeta + polarization
     - High accuracy, expensive
   * - QZP
     - Quadruple-zeta + polarization
     - Benchmarking only

**Cost Scaling**:

* SZ → DZ: ~2× slower
* DZ → DZP: ~1.5× slower (add polarization)
* DZP → TZP: ~2× slower
* Total: TZP is ~6× slower than SZ

**Recommended Production Parameters**:

.. code-block:: python

   # Standard accuracy (recommended for most work)
   user_params = {
       "PAO.BasisSize": "DZP",
       "PAO.EnergyShift": "0.01 Ry",
       "PAO.SplitNorm": 0.15,
   }

   # High accuracy (expensive, for benchmarking)
   user_params = {
       "PAO.BasisSize": "TZP",
       "PAO.EnergyShift": "0.005 Ry",
       "PAO.SplitNorm": 0.20,
   }

📁 **Location**: ``tutorials/02-convergence/03-complete-basis/``

⏱️ **Time**: 90 minutes (simple: 30 min, full: 1-4 hours)

⭐ **Difficulty**: Advanced

----

Convergence Best Practices
===========================

General Guidelines
------------------

1. **Test one parameter at a time**

   * Keep other parameters fixed
   * Easier to analyze and debug

2. **Start with coarse sampling, refine as needed**

   * K-points: 2×2×2, 4×4×4, 6×6×6, ...
   * Cutoff: 200, 300, 400, 500 Ry

3. **Plot results**

   * Energy vs. parameter value
   * Look for plateau or asymptotic behavior

4. **Check multiple properties**

   * Total energy (most sensitive)
   * Forces (for relaxations)
   * Band gap (for semiconductors)
   * Lattice parameters (for structure prediction)

5. **Document converged parameters**

   * Include in all production scripts
   * Note in publications

6. **Use conservative criteria**

   * Aim for < 1 meV/atom for energies
   * Tighter for energy differences (e.g., formation energies)

Recommended Order
-----------------

For new material:

1. **Mesh cutoff** (quickest to converge)
2. **K-points** (material-dependent)
3. **Basis size** (largest effect)
4. **PAO.EnergyShift** (fine-tuning)
5. **PAO.SplitNorm** (usually least important)

Material-Specific Considerations
---------------------------------

**Metals**:

* Denser k-points (Fermi surface)
* MP or FD occupation function
* Electronic temperature (~300-1000 K)

**Semiconductors/Insulators**:

* Standard k-points sufficient
* Default occupation function
* No smearing needed

**2D Materials**:

* Dense in-plane k-points
* Γ-point in out-of-plane direction
* Large vacuum spacing (> 15 Å)

**Molecules**:

* Γ-point only
* Focus on basis quality
* Large supercell to avoid image interactions

----

Next Steps
==========

After converging parameters:

1. Apply to **production workflows** (Tutorials 09-18)
2. Consider **tier presets** for material-specific settings (Tutorial 18)
3. Enable **error handling** for robust calculations (Tutorial 15)

See :doc:`index` for full tutorial listing.

----

.. note::

   Convergence testing is a one-time investment that ensures all subsequent
   calculations are reliable and reproducible. Don't skip this step!
