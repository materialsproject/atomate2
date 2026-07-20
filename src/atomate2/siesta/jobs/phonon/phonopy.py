"""Phonon calculation jobs using phonopy with SIESTA."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Maker, Response, job
from pymatgen.core import Structure

from atomate2.siesta.jobs.core import StaticMaker

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PhonopyMaker(Maker):
    """
    Maker for phonon calculations using phonopy with SIESTA.

    This workflow automates phonon calculations by:
    1. Optionally relaxing the input structure
    2. Generating an appropriate supercell
    3. Creating symmetry-reduced atomic displacements
    4. Running SIESTA force calculations for each displacement
    5. Using phonopy to compute phonon properties from forces

    Parameters
    ----------
    name : str
        Name of the workflow
    supercell_matrix : list[list[int]] | None
        Supercell matrix for phonon calculations. If None, will be
        automatically determined based on min_length.
    displacement : float
        Atomic displacement distance in Angstroms (default: 0.01)
    symprec : float
        Symmetry precision for reducing displacements (default: 1e-5)
    relax_maker : Maker | None
        Maker for initial structure relaxation. If None, assumes
        structure is already relaxed.
    static_maker : Maker
        Maker for force calculations on displaced structures
    min_length : float | None
        Minimum supercell lattice vector length in Angstroms.
        Used to auto-generate supercell_matrix if not provided.
    prefer_90_degrees : bool
        When auto-generating supercell, prefer matrices that keep
        angles close to 90 degrees (default: True)
    use_symmetry : bool
        Use crystal symmetry to reduce number of displacements (default: True)
    create_thermal_properties : bool
        Calculate thermal properties (Cv, entropy, free energy) (default: True)
    t_step : float
        Temperature step for thermal properties in K (default: 10)
    t_max : float
        Maximum temperature for thermal properties in K (default: 1000)
    t_min : float
        Minimum temperature for thermal properties in K (default: 0)
    mesh : tuple[int, int, int]
        q-point mesh for DOS and thermal properties (default: (50, 50, 50))

    Examples
    --------
    >>> from atomate2.siesta.jobs.phonon import PhonopyMaker
    >>> from pymatgen.core import Structure
    >>> structure = Structure.from_file("POSCAR")
    >>> maker = PhonopyMaker(min_length=15.0)
    >>> flow = maker.make(structure)
    """

    name: str = "phonopy"
    supercell_matrix: list[list[int]] | None = None
    displacement: float = 0.01
    symprec: float = 1e-5
    relax_maker: Maker | None = None
    static_maker: Maker = field(default_factory=StaticMaker)
    min_length: float | None = None
    prefer_90_degrees: bool = True
    use_symmetry: bool = True
    create_thermal_properties: bool = True
    t_step: float = 10
    t_max: float = 1000
    t_min: float = 0
    mesh: tuple[int, int, int] = (50, 50, 50)

    # Dry-run support
    dry_run: bool = False
    dry_run_output_dir: str = "dry_run_output"
    dry_run_format: str = "cif"

    def __post_init__(self):
        """Propagate dry-run settings to child makers."""
        if self.dry_run:
            # Propagate to relax_maker
            if self.relax_maker and hasattr(self.relax_maker, "dry_run"):
                self.relax_maker.dry_run = True
                self.relax_maker.dry_run_output_dir = self.dry_run_output_dir
                self.relax_maker.dry_run_format = self.dry_run_format
            # Propagate to static_maker
            if self.static_maker and hasattr(self.static_maker, "dry_run"):
                self.static_maker.dry_run = True
                self.static_maker.dry_run_output_dir = self.dry_run_output_dir
                self.static_maker.dry_run_format = self.dry_run_format

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        supercell_matrix: list[list[int]] | None = None,
    ) -> Flow:
        """
        Create phonon calculation workflow.

        Parameters
        ----------
        structure : Structure
            Input structure (should be relaxed or will be relaxed if relax_maker provided)
        prev_dir : str | Path | None
            Previous directory for reusing files
        supercell_matrix : list[list[int]] | None
            Override the supercell matrix for this specific calculation.
            If provided, overrides self.supercell_matrix for this make() call only.

        Returns
        -------
        Flow
            Phonon calculation workflow
        """
        jobs = []

        # Override supercell_matrix if provided as argument
        # This allows QHA workflows to specify different supercells per volume
        if supercell_matrix is not None:
            effective_supercell_matrix = supercell_matrix
        else:
            effective_supercell_matrix = self.supercell_matrix

        # Step 1: Optional structure relaxation
        if self.relax_maker is not None:
            logger.info("Adding structure relaxation step")
            relax = self.relax_maker.make(structure, prev_dir=prev_dir)
            relax.name = f"{self.name}_relax"
            jobs.append(relax)
            structure = relax.output.structure
            prev_dir = relax.output.dir_name

        # Step 2: Generate supercell and displacements
        logger.info("Generating displacements")
        displacement_job = generate_phonon_displacements(
            structure=structure,
            supercell_matrix=effective_supercell_matrix,
            displacement=self.displacement,
            symprec=self.symprec,
            min_length=self.min_length,
            prefer_90_degrees=self.prefer_90_degrees,
            use_symmetry=self.use_symmetry,
        )
        displacement_job.name = f"{self.name}_generate_displacements"
        jobs.append(displacement_job)

        # Step 3: Calculate forces for each displacement
        logger.info("Creating force calculation jobs")
        force_job = run_force_calculations(
            displaced_structures=displacement_job.output["displaced_structures"],
            static_maker=self.static_maker,
            prev_dir=prev_dir,
        )
        force_job.name = f"{self.name}_calculate_forces"
        jobs.append(force_job)

        # Step 4: Phonopy analysis
        logger.info("Adding phonopy analysis job")
        phonopy_job = run_phonopy_analysis(
            structure=structure,
            phonopy_settings=displacement_job.output["phonopy_settings"],
            forces=force_job.output["forces"],
            energies=force_job.output["energies"],
            supercell_matrix=displacement_job.output["supercell_matrix"],
            displacement=self.displacement,
            symprec=self.symprec,
            create_thermal_properties=self.create_thermal_properties,
            mesh=self.mesh,
            t_step=self.t_step,
            t_max=self.t_max,
            t_min=self.t_min,
        )
        phonopy_job.name = f"{self.name}_analysis"
        jobs.append(phonopy_job)

        # Note: Plots and summary are generated directly within the phonopy_job
        # (see run_phonopy_analysis function). This avoids creating extra empty job folders.

        return Flow(jobs, output=phonopy_job.output, name=self.name)


@job
def generate_phonon_displacements(
    structure: Structure,
    supercell_matrix: list[list[int]] | None = None,
    displacement: float = 0.01,
    symprec: float = 1e-5,
    min_length: float | None = None,
    prefer_90_degrees: bool = True,
    use_symmetry: bool = True,
) -> dict[str, Any]:
    """
    Generate phonon displacements using phonopy.

    Parameters
    ----------
    structure : Structure
        Input structure
    supercell_matrix : list[list[int]] | None
        Supercell matrix. If None, auto-generated from min_length.
    displacement : float
        Displacement distance in Angstroms
    symprec : float
        Symmetry precision
    min_length : float | None
        Minimum supercell length
    prefer_90_degrees : bool
        Prefer 90 degree angles in supercell
    use_symmetry : bool
        Use symmetry to reduce displacements

    Returns
    -------
    dict
        Dictionary containing:
        - displaced_structures: list of displaced Structure objects
        - supercell_matrix: the supercell matrix used
        - phonopy_settings: dict with phonopy settings for later use
    """
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    logger.info(f"Generating phonon displacements for {structure.formula}")
    logger.info(
        f"  supercell_matrix={supercell_matrix}, min_length={min_length}, "
        f"structure.num_sites={structure.num_sites}"
    )

    # Convert pymatgen structure to phonopy atoms
    phonopy_atom = PhonopyAtoms(
        symbols=[str(s) for s in structure.species],
        cell=structure.lattice.matrix,
        scaled_positions=structure.frac_coords,
    )

    # Determine supercell matrix
    if supercell_matrix is None:
        if min_length is None:
            min_length = 10.0
            logger.warning(
                f"No supercell_matrix or min_length specified, using min_length={min_length} Å"
            )

        supercell_matrix = _get_supercell_matrix(
            structure, min_length, prefer_90_degrees
        )
        logger.info(f"Auto-generated supercell matrix: {supercell_matrix}")

    # Calculate expected supercell size
    import numpy as np

    sc_matrix = np.array(supercell_matrix)
    sc_mult = int(round(abs(np.linalg.det(sc_matrix))))
    expected_atoms = structure.num_sites * sc_mult
    logger.info(
        f"Final supercell: {structure.num_sites} atoms × {sc_mult} = {expected_atoms} atoms"  # noqa: RUF001
    )

    # Create Phonopy object
    phonon = Phonopy(
        phonopy_atom,
        supercell_matrix=supercell_matrix,
        symprec=symprec,
        is_symmetry=use_symmetry,
    )

    # Generate displacements
    phonon.generate_displacements(distance=displacement)

    # Get displaced supercells
    supercells_with_disps = phonon.supercells_with_displacements

    # Convert back to pymatgen structures
    displaced_structures = []
    for supercell in supercells_with_disps:
        # Create Structure directly from phonopy supercell
        # supercell.cell is the lattice matrix (numpy array)
        # supercell.symbols is the list of element symbols
        # supercell.positions is cartesian coordinates (numpy array)
        pmg_structure = Structure(
            lattice=supercell.cell,
            species=supercell.symbols,
            coords=supercell.positions,
            coords_are_cartesian=True,
        )
        displaced_structures.append(pmg_structure)

    logger.info(
        f"Generated {len(displaced_structures)} displaced structures "
        f"(symmetry: {use_symmetry})"
    )

    # Store phonopy settings for later analysis
    phonopy_settings = {
        "supercell_matrix": supercell_matrix,
        "displacement": displacement,
        "symprec": symprec,
        "use_symmetry": use_symmetry,
    }

    return {
        "displaced_structures": displaced_structures,
        "supercell_matrix": supercell_matrix,
        "phonopy_settings": phonopy_settings,
    }


@job
def run_force_calculations(
    displaced_structures: list[Structure],
    static_maker: Maker,
    prev_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run SIESTA force calculations for displaced structures.

    Parameters
    ----------
    displaced_structures : list[Structure]
        List of displaced structures
    static_maker : Maker
        Maker for static calculations
    prev_dir : str | Path | None
        Previous directory for file reuse

    Returns
    -------
    dict
        Dictionary containing:
        - forces: list of force arrays for each displacement
        - energies: list of energies for each displacement
    """
    logger.info(
        f"Running force calculations for {len(displaced_structures)} displacements"
    )

    force_jobs = []
    for i, struct in enumerate(displaced_structures):
        force_job = static_maker.make(struct, prev_dir=prev_dir)
        force_job.name = f"displacement_{i + 1}_of_{len(displaced_structures)}"
        force_jobs.append(force_job)

    # Create a flow to run all force calculations
    force_flow = Flow(force_jobs, name="force_calculations")

    # Collect forces and energies
    forces = [job.output.output.forces for job in force_jobs]
    energies = [job.output.output.energy for job in force_jobs]

    return Response(
        output={"forces": forces, "energies": energies},
        detour=force_flow,
    )


@job
def run_phonopy_analysis(
    structure: Structure,
    phonopy_settings: dict,
    forces: list[np.ndarray],
    energies: list[float],
    supercell_matrix: list[list[int]],
    displacement: float,
    symprec: float,
    create_thermal_properties: bool = True,
    mesh: tuple[int, int, int] = (50, 50, 50),
    t_step: float = 10,
    t_max: float = 1000,
    t_min: float = 0,
    filename_phonopy_yaml: str = "phonopy.yaml",
) -> dict[str, Any]:
    """
    Run phonopy analysis on collected forces.

    Parameters
    ----------
    structure : Structure
        Original structure
    phonopy_settings : dict
        Settings from displacement generation
    forces : list[np.ndarray]
        Forces for each displacement
    energies : list[float]
        DFT energies for each displacement
    supercell_matrix : list[list[int]]
        Supercell matrix
    displacement : float
        Displacement distance
    symprec : float
        Symmetry precision
    create_thermal_properties : bool
        Calculate thermal properties
    mesh : tuple[int, int, int]
        q-point mesh
    t_step : float
        Temperature step in K
    t_max : float
        Max temperature in K
    t_min : float
        Min temperature in K
    filename_phonopy_yaml : str
        Filename for saving phonopy.yaml (default: "phonopy.yaml").
        Gruneisen workflows use ground_phonopy.yaml, plus_phonopy.yaml, minus_phonopy.yaml

    Returns
    -------
    dict
        Complete phonon analysis results
    """
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    logger.info("Running phonopy analysis")
    logger.info(f"Received {len(forces)} force arrays")

    # Convert structure to phonopy atoms
    phonopy_atom = PhonopyAtoms(
        symbols=[str(s) for s in structure.species],
        cell=structure.lattice.matrix,
        scaled_positions=structure.frac_coords,
    )

    # Create Phonopy object
    phonon = Phonopy(
        phonopy_atom,
        supercell_matrix=supercell_matrix,
        symprec=symprec,
        is_symmetry=phonopy_settings["use_symmetry"],
    )

    # Generate displacements (needed to set up phonopy)
    phonon.generate_displacements(distance=displacement)

    # Convert forces to proper numpy array format
    # Forces should be a list of arrays, one per displacement
    # Each array should have shape (n_atoms, 3)
    forces_array = []
    for i, force_set in enumerate(forces):
        if force_set is None:
            raise ValueError(
                f"Forces for displacement {i} are None! Check SIESTA calculation output."
            )

        # Convert to numpy array if needed
        if not isinstance(force_set, np.ndarray):
            force_set = np.array(force_set)

        # Ensure correct shape
        if force_set.ndim != 2 or force_set.shape[1] != 3:
            raise ValueError(
                f"Forces for displacement {i} have wrong shape: {force_set.shape}. "
                f"Expected (n_atoms, 3)"
            )

        forces_array.append(force_set)
        logger.info(f"Displacement {i}: forces shape = {force_set.shape}")

    # Set forces as numpy array
    forces_np = np.array(forces_array)
    logger.info(f"Final forces array shape: {forces_np.shape}")
    phonon.forces = forces_np

    # Produce force constants
    phonon.produce_force_constants()

    # Save phonopy.yaml file for Gruneisen and other workflows
    # The filename will be set by the Gruneisen workflow to ground_phonopy.yaml,
    # plus_phonopy.yaml, or minus_phonopy.yaml

    # Get current working directory and construct full path
    cwd = Path.cwd().resolve()

    # Save to current directory (will be the job's directory when jobflow runs it)
    phonon.save(filename_phonopy_yaml, settings={"force_constants": True})

    # Construct the full path where the file was saved
    yaml_path = (cwd / filename_phonopy_yaml).resolve()

    # Verify file was created
    if not yaml_path.exists():
        # Fallback: try alternative resolution
        alt_path = Path(filename_phonopy_yaml).resolve()
        if alt_path.exists():
            yaml_path = alt_path

    # Store the FULL PATH to the yaml file (not just directory)
    # This is crucial for SIESTA's Gruneisen workflow
    yaml_file_path = str(yaml_path)
    logger.info(f"Phonopy yaml saved to: {yaml_file_path}")

    # Get phonon properties
    results = {
        "structure": structure,
        "supercell_matrix": supercell_matrix,
        "displacement": displacement,
        "symprec": symprec,
        "n_displacements": len(forces),
        "force_constants": phonon.force_constants.tolist(),
    }

    # Calculate phonon frequencies and check for imaginary modes
    phonon.run_mesh(mesh)
    mesh_dict = phonon.get_mesh_dict()
    frequencies = mesh_dict["frequencies"]

    results["has_imaginary_frequencies"] = bool(np.any(frequencies < -0.01))
    results["min_frequency"] = float(np.min(frequencies))
    results["max_frequency"] = float(np.max(frequencies))

    # Store DFT energy for QHA calculations
    # Use the average energy from displacement calculations as the reference energy
    if energies is not None and len(energies) > 0:
        # Filter out None values
        valid_energies = [e for e in energies if e is not None]
        if valid_energies:
            avg_energy = np.mean(valid_energies)
            # Calculate per-formula-unit energy
            formula_units = (
                structure.composition.num_atoms
                / structure.composition.reduced_composition.num_atoms
            )
            results["total_dft_energy"] = float(avg_energy / formula_units)
            logger.info(
                f"Stored total_dft_energy={results['total_dft_energy']} eV/f.u."
            )
        else:
            logger.warning("All energies are None, total_dft_energy will not be set")
    else:
        logger.warning("No energies provided, total_dft_energy will not be set")

    # Calculate thermal properties
    if create_thermal_properties:
        logger.info("Calculating thermal properties")
        phonon.run_thermal_properties(t_step=t_step, t_max=t_max, t_min=t_min)
        tp_dict = phonon.get_thermal_properties_dict()

        results["thermal_properties"] = {
            "temperatures": tp_dict["temperatures"].tolist(),
            "free_energy": tp_dict["free_energy"].tolist(),
            "entropy": tp_dict["entropy"].tolist(),
            "heat_capacity": tp_dict["heat_capacity"].tolist(),
        }

    logger.info(
        f"Phonopy analysis complete. Imaginary modes: {results['has_imaginary_frequencies']}"
    )

    # Store the full path to the phonopy.yaml file for Gruneisen workflows
    # SIESTA uses full file paths rather than directory paths
    results["dir_name"] = yaml_file_path

    # Generate plots and summary in the same directory as phonopy.yaml
    logger.info("Generating phonon plots and summary")
    from atomate2.siesta.jobs.phonon.plotting import (
        plot_phonon_band_structure,
        plot_phonon_dos,
        plot_thermal_properties,
        write_phonon_summary,
    )

    output_dir = str(Path(yaml_file_path).parent)

    # Call the underlying functions (without job decorator) using .original
    # This avoids creating separate job objects
    try:
        plot_phonon_band_structure.original(phonon_doc=results, output_dir=output_dir)
        logger.info("Generated phonon band structure plot")
    except Exception as e:
        logger.warning(f"Failed to generate band structure plot: {e}")

    try:
        plot_phonon_dos.original(phonon_doc=results, output_dir=output_dir)
        logger.info("Generated phonon DOS plot")
    except Exception as e:
        logger.warning(f"Failed to generate DOS plot: {e}")

    if create_thermal_properties:
        try:
            plot_thermal_properties.original(phonon_doc=results, output_dir=output_dir)
            logger.info("Generated thermal properties plot")
        except Exception as e:
            logger.warning(f"Failed to generate thermal properties plot: {e}")

    try:
        write_phonon_summary.original(phonon_doc=results, output_dir=output_dir)
        logger.info("Generated phonon summary file")
    except Exception as e:
        logger.warning(f"Failed to generate summary file: {e}")

    return results


def _get_supercell_matrix(
    structure: Structure, min_length: float, prefer_90_degrees: bool = True
) -> list[list[int]]:
    """
    Determine supercell matrix based on minimum length requirement.

    Parameters
    ----------
    structure : Structure
        Input structure
    min_length : float
        Minimum supercell length in Angstroms
    prefer_90_degrees : bool
        Prefer supercells with 90 degree angles

    Returns
    -------
    list[list[int]]
        Supercell matrix
    """
    lattice = structure.lattice
    abc = lattice.abc

    # Calculate minimum multipliers for each direction
    multipliers = [int(np.ceil(min_length / length)) for length in abc]

    # For non-orthogonal cells, we may need larger multipliers
    if prefer_90_degrees and not lattice.is_orthogonal:
        # Use slightly larger multipliers for non-orthogonal cells
        multipliers = [m + 1 if m < 3 else m for m in multipliers]

    # Create diagonal supercell matrix
    supercell_matrix = [
        [multipliers[0], 0, 0],
        [0, multipliers[1], 0],
        [0, 0, multipliers[2]],
    ]

    return supercell_matrix
