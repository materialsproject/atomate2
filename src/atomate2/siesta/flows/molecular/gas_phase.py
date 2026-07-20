"""Gas-phase molecular calculation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, job

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker

if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure

    from atomate2.siesta.jobs.base import BaseSiestaMaker


@dataclass
class GasPhaseMoleculeMaker(BaseSiestaFlowMaker):
    """
    General-purpose gas-phase molecular calculation workflow.

    Performs geometry optimization and energy calculation for isolated molecules
    in a large vacuum box. Essential for obtaining reference energies for
    thermochemical calculations, reaction energies, and molecular properties.

    **General Applications** (not limited to electrocatalysis):
    - Reference energies for any reaction (O₂, H₂O, CO₂, H₂, N₂, etc.)
    - Molecular thermochemistry (ΔH, ΔG, ΔS)
    - Bond dissociation energies
    - Molecular properties (dipole, polarizability)
    - Combustion reactions
    - Organic chemistry reactions
    - Atmospheric chemistry

    **Electrocatalysis Applications**:
    - ORR references: O₂, H₂O
    - OER references: O₂, H₂O
    - HER references: H₂, (H₂O optional)
    - CO₂RR references: CO₂, CO, CH₄, C₂H₄, etc.
    - Metal-air batteries: O₂, bulk products (Li₂O₂, Na₂O₂, etc.)

    Workflow Steps:
    ---------------
    1. Place molecule in large vacuum box (no periodic boundary effects)
    2. Optional: Set up spin-polarized calculation for open-shell molecules
    3. Optimize molecular geometry
    4. Calculate final energy and properties

    **Spin Handling**:
    ------------------
    For paramagnetic molecules (O₂, O, OH, NO), spin polarization is CRITICAL:
    - O₂: Triplet ground state (S=1), requires spin polarization
    - O: Triplet ground state (S=1)
    - OH: Doublet ground state (S=1/2)
    - H₂O, CO₂, N₂, CO, H₂: Closed-shell singlet (S=0)

    Use `spin_polarized=True` for open-shell species or SIESTA will converge
    to the wrong (closed-shell) state!

    **Automatic Spin Detection**:
    - Set `spin_polarized=None` (default) for automatic detection
    - Uses `get_siesta_spin_config()` to determine spin state
    - Supports common molecules: O₂, O, OH, NO, H₂O, CO₂, etc.

    Parameters
    ----------
    name : str
        Name for the workflow
    relax_maker : BaseSiestaMaker
        The maker to use for geometry optimization (default: RelaxMaker)
    spin_polarized : bool | None
        Whether to use spin-polarized DFT:
        - True: Force spin polarization (for O₂, O, OH, NO, etc.)
        - False: Non-polarized (for H₂O, CO₂, N₂, H₂, etc.)
        - None: Auto-detect based on molecular formula (default)
    box_size : float
        Size of cubic vacuum box (Å). Default: 15.0 Å
        Increase for large molecules or dipole moment calculations
    user_params : dict | None
        Additional SIESTA parameters to override defaults
        Example: {"PAO.BasisSize": "TZP", "Mesh.Cutoff": "400 Ry"}

    Examples
    --------
    >>> from pymatgen.core import Molecule
    >>> from atomate2.siesta.flows.molecular import GasPhaseMoleculeMaker
    >>> from jobflow import run_locally
    >>>
    >>> # Example 1: O₂ molecule (paramagnetic, auto-detect spin)
    >>> o2 = Molecule(["O", "O"], [[0, 0, 0], [0, 0, 1.21]])
    >>> maker = GasPhaseMoleculeMaker()  # spin_polarized=None (auto-detect)
    >>> flow = maker.make(o2)
    >>> result = run_locally(flow, create_folders=True)
    >>>
    >>> # Example 2: H₂O molecule (closed-shell, explicit)
    >>> h2o = Molecule.from_file("h2o.xyz")
    >>> maker = GasPhaseMoleculeMaker(spin_polarized=False)
    >>> flow = maker.make(h2o)
    >>>
    >>> # Example 3: With custom parameters
    >>> co2 = Molecule.from_file("co2.xyz")
    >>> maker = GasPhaseMoleculeMaker(
    ...     spin_polarized=False,
    ...     box_size=20.0,  # Larger box
    ...     user_params={"PAO.BasisSize": "TZP", "Mesh.Cutoff": "400 Ry"},
    ... )
    >>> flow = maker.make(co2)
    >>>
    >>> # Example 4: Force spin polarization for radical
    >>> oh = Molecule(["O", "H"], [[0, 0, 0], [0, 0, 0.97]])
    >>> maker = GasPhaseMoleculeMaker(spin_polarized=True)
    >>> flow = maker.make(oh)

    Returns
    -------
    Flow
        Jobflow Flow object containing the molecular calculation

    Notes
    -----
    - Molecules are placed in a cubic box with periodic boundary conditions
    - Box size should be large enough to avoid self-interaction (default: 15 Å)
    - For accurate dipole moments, use box_size ≥ 20 Å
    - Spin-polarized calculations take ~2× longer but are essential for radicals
    - Results include: total energy, optimized geometry, forces, dipole moment

    See Also
    --------
    atomate2.siesta.flows.electrocatalysis.utils.spin_config.get_siesta_spin_config :
        Automatic spin configuration for common molecules
    """  # noqa: RUF002

    name: str = "gas_phase_molecule"
    relax_maker: BaseSiestaMaker = field(default_factory=RelaxMaker)
    spin_polarized: bool | None = None  # None = auto-detect
    box_size: float = 15.0  # Å
    user_params: dict | None = None

    def make(self, molecule: Molecule) -> Flow:
        """
        Create gas-phase molecular calculation workflow.

        Parameters
        ----------
        molecule : Molecule
            Pymatgen Molecule object to calculate

        Returns
        -------
        Flow
            Complete workflow for gas-phase calculation
        """
        jobs = []

        # 1. Prepare molecule in box
        molecule_in_box_job = _prepare_molecule_in_box(
            molecule=molecule,
            box_size=self.box_size,
        )
        jobs.append(molecule_in_box_job)

        # 2. Detect spin configuration (inline, not via job)
        from atomate2.siesta.flows.electrocatalysis.utils.spin_config import (
            get_siesta_spin_config,
        )

        formula = molecule.composition.reduced_formula
        spin_config = get_siesta_spin_config(formula)

        if self.spin_polarized is None:
            # Auto-detect from molecular formula
            spin_polarized = spin_config["spin_polarized"]
        else:
            # User-specified
            spin_polarized = self.spin_polarized

        # 3. Set up parameters
        params = self.user_params.copy() if self.user_params else {}

        if spin_polarized:
            # Set Spin parameter
            params["Spin"] = "polarized"

            # IMPORTANT: Do NOT use Spin.Fix for molecules during relaxation!
            # Spin.Fix + CG causes SIESTA to crash. DM.InitSpin is sufficient.
            # Only set Spin.Total as a guide (not strict constraint)
            if spin_config.get("fix_spin", False):
                # params["Spin.Fix"] = True  # DISABLED: causes SIESTA crash with CG
                params["Spin.Total"] = spin_config["total_spin_moment"]

        # 4. Apply magnetic moments to structure if needed
        if spin_polarized and spin_config.get("init_magnetic_moments") is not None:
            # Preserve the exact per-atom moments applied below. The DM.InitSpin
            # auto-generator defaults to antiferromagnetic ordering, which flips
            # signs by atom index (e.g. O2 +1/+1 -> +1/-1 -> net S=0 SINGLET,
            # 0.44 eV above the true triplet). "custom" keeps the molecular
            # moments as-is (+1/+1 -> net S=2 triplet). User override respected.
            params.setdefault("a2s_magnetic_ordering", "custom")

            # Apply magmoms via job
            magmom_job = _apply_magnetic_moments(
                structure=molecule_in_box_job.output.structure,
                magmom_dict=spin_config["init_magnetic_moments"],
            )
            jobs.append(magmom_job)
            input_structure = magmom_job.output.structure
        else:
            # No magnetic moments needed
            input_structure = molecule_in_box_job.output.structure

        # 5. Run relaxation
        relax_job = self.relax_maker.make(
            input_structure,
            prev_dir=None,
        )
        relax_job.name = f"{self.name}_{molecule.composition.reduced_formula}"

        # Update parameters if needed
        if params:
            from atomate2.siesta.powerups import update_user_siesta_settings

            relax_job = update_user_siesta_settings(relax_job, params)

        jobs.append(relax_job)

        # 6. Extract results
        results_job = _extract_molecular_results(
            molecule_formula=molecule.composition.reduced_formula,
            total_energy=relax_job.output.output.energy,
            structure=relax_job.output.structure,
            spin_polarized=spin_polarized,
            spin_config=spin_config if self.spin_polarized is None else None,
        )
        jobs.append(results_job)

        return Flow(jobs, output=results_job.output, name=self.name)


@job
def _prepare_molecule_in_box(molecule: Molecule, box_size: float) -> dict:
    """
    Place molecule in cubic box with periodic boundary conditions.

    Parameters
    ----------
    molecule : Molecule
        Molecule to place in box
    box_size : float
        Size of cubic box (Å)

    Returns
    -------
    dict
        {"structure": Structure (molecule in box)}
    """
    from pymatgen.core import Lattice, Structure

    # Center molecule at origin
    molecule_centered = molecule.get_centered_molecule()

    # Create cubic lattice
    lattice = Lattice.cubic(box_size)

    # Convert molecule to Structure (in box center)
    structure = Structure(
        lattice,
        molecule_centered.species,
        molecule_centered.cart_coords,
        coords_are_cartesian=True,
    )

    return {"structure": structure}


@job
def _apply_magnetic_moments(structure: Structure, magmom_dict: dict) -> dict:
    """
    Apply initial magnetic moments to structure for spin-polarized calculations.

    This is critical for open-shell molecules (O₂, O, OH) - without initial
    magnetic moments, SIESTA may converge to the wrong spin state!

    Parameters
    ----------
    structure : Structure
        Structure to modify (molecule in box).
    magmom_dict : dict
        Element-to-moment mapping, e.g., {"O": 1.0}.
        Each atom of element "O" gets 1.0 μB initial moment.

    Returns
    -------
    dict
        {"structure": Structure with magmom site property}

    Notes
    -----
    For O₂ with magmom_dict={"O": 1.0}:
        - 2 O atoms each get 1.0 μB → total 2.0 μB (triplet state)
        - SIESTA converts to DM.InitSpin block automatically
    """
    structure_copy = structure.copy()

    # Build list of magnetic moments for each site
    magmoms = []
    for site in structure_copy:
        element = site.specie.symbol
        magmom = magmom_dict.get(element, 0.0)
        magmoms.append(magmom)

    # Add as site property
    structure_copy.add_site_property("magmom", magmoms)

    return {"structure": structure_copy}


@job
def _extract_molecular_results(
    molecule_formula: str,
    total_energy: float,
    structure: Structure,
    spin_polarized: bool,
    spin_config: dict | None = None,
) -> dict:
    """
    Extract and format molecular calculation results.

    Parameters
    ----------
    molecule_formula : str
        Chemical formula
    total_energy : float
        Total energy (eV)
    structure : Structure
        Optimized structure
    spin_polarized : bool
        Whether calculation was spin-polarized
    spin_config : dict | None
        Spin configuration details (if auto-detected)

    Returns
    -------
    dict
        GasPhaseMoleculeDocument-compatible dictionary
    """
    # Convert structure back to molecule
    from pymatgen.core import Molecule

    molecule = Molecule.from_sites(structure.sites)

    result = {
        "formula": molecule_formula,
        "total_energy": total_energy,
        "spin_polarized": spin_polarized,
        "spin_type": "polarized" if spin_polarized else "non-polarized",
        "structure": structure,
        "molecule": molecule,
        "composition": molecule.composition,
    }

    if spin_config:
        result["spin_config"] = spin_config

    return result
