"""Schemas for the finite-temperature phonon workflow."""

from typing import Literal

from emmet.core.math import Matrix3D
from emmet.core.structure import StructureMetadata
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos

from atomate2.common.schemas.phonons import ForceConstants, PhononJobDirs, PhononUUIDs


class TrajectoryHealth(BaseModel):
    """Check whether the MD trajectory stayed at the reference structure.

    A second-order fit describes small vibrations around the reference
    structure. If the atoms melt, move to another structure or diffuse, the
    fitted force constants may not describe small vibrations around the
    reference. The fit error does not show this, so it is measured on the
    trajectory. The displacement of the center of mass of each frame is
    removed from its displacements.
    """

    verdict: (
        Literal[
            "stable",
            "melted",
            "transformed",
            "shifted_or_diffusing",
            "disordering",
            "diffusing_or_soft",
            "reference_mismatch",
        ]
        | None
    ) = Field(
        None,
        description="Result of the check. 'reference_mismatch' means the atoms of "
        "the MD are not in the order of the reference. Every other verdict except "
        "'stable' means the trajectory left the reference structure.",
    )
    n_frames: int | None = Field(None, description="Number of MD frames analysed.")
    rms_displacement_start: float | None = Field(
        None,
        description="Root mean square displacement from the reference structure in "
        "the first 10 frames, in Angstrom. The MD starts from the reference, so a "
        "large value points at a mismatch in atom order.",
    )
    rms_displacement_end: float | None = Field(
        None,
        description="Root mean square displacement from the reference structure in "
        "the last fifth of the frames, in Angstrom.",
    )
    u_ref: float | None = Field(
        None,
        description="Root mean square displacement from the reference structure in "
        "the second half of the frames, in Angstrom.",
    )
    u_shift: float | None = Field(
        None,
        description="Root mean square displacement of the mean atomic positions from "
        "the reference structure in the second half of the frames, in Angstrom.",
    )
    u_vib: float | None = Field(
        None,
        description="Root mean square vibration around the mean atomic positions in "
        "the second half of the frames, in Angstrom. u_ref**2 = u_shift**2 + "
        "u_vib**2.",
    )
    nearest_neighbor_distance: float | None = Field(
        None,
        description="Nearest-neighbor distance of each atom in the reference "
        "structure, averaged over the atoms, in Angstrom.",
    )
    lindemann_ratio: float | None = Field(
        None, description="u_vib divided by the nearest-neighbor distance."
    )
    shift_ratio: float | None = Field(None, description="u_ref divided by u_vib.")
    energy_drift: float | None = Field(
        None,
        description="Mean potential energy in the last quarter of the frames minus "
        "that in the second quarter, in eV/atom. None for fewer than 20 frames.",
    )
    energy_fluctuation: float | None = Field(
        None,
        description="Standard deviation of the potential energy in the last fifth of "
        "the frames, in eV/atom.",
    )


class FiniteTemperaturePhononDoc(StructureMetadata):
    """Effective harmonic phonons at a finite temperature."""

    structure: Structure | None = Field(
        None,
        description="Unit cell of the NVT MD and the fit, with its atoms sorted by "
        "electronegativity. It is the relaxed unit cell, or the cell from the NPT MD "
        "if there was one.",
    )
    temperature: float | None = Field(None, description="MD temperature in K.")
    code: str | None = Field(
        None,
        description="Code of the phonon displacement calculations that give the "
        "forces, 'vasp' or 'forcefields'.",
    )
    md_code: str | None = Field(
        None, description="Code of the MD, 'vasp' or 'forcefields'."
    )
    force_field_name: str | None = Field(
        None,
        description="Force field of the phonon displacement calculations, if a "
        "force field was used.",
    )
    force_field_kwargs: dict | None = Field(
        None,
        description="Keyword arguments of the force field calculator of the phonon "
        "displacement calculations.",
    )
    md_force_field_name: str | None = Field(
        None, description="Force field of the MD, if a force field was used."
    )
    md_force_field_kwargs: dict | None = Field(
        None, description="Keyword arguments of the force field calculator of the MD."
    )
    thermostat: str | None = Field(None, description="Thermostat of the NVT MD.")
    md_time_step: float | None = Field(None, description="MD time step in fs.")
    md_time: float | None = Field(
        None,
        description="Length of the joined MD trajectory, the number of frames times "
        "the time step, in ps.",
    )
    equilibration_time: float | None = Field(
        None, description="Time at the start of the trajectory left out, in ps."
    )
    n_snapshots: int | None = Field(
        None, description="Number of MD snapshots in the force constant fit."
    )
    snapshot_times: list[float] | None = Field(
        None,
        description="MD time of each snapshot in ps, its step number times the time "
        "step.",
    )
    supercell_matrix: Matrix3D | None = Field(
        None, description="Supercell matrix of the MD and the statics."
    )
    primitive_matrix: Matrix3D | None = Field(
        None, description="Primitive matrix used by phonopy."
    )
    rotational_sum_rule: str | None = Field(
        None, description="Rotational sum rule passed to pheasy with --rasr."
    )
    lasso_alpha: float | None = Field(
        None, description="LASSO penalty chosen by cross-validation in pheasy."
    )
    force_rmse: float | None = Field(
        None,
        description="Root mean square error of the symmetrized force constants on "
        "the snapshot forces used in the fit, in eV/Angstrom.",
    )
    max_residual_force: float | None = Field(
        None,
        description="Largest force component on the undisplaced supercell, in "
        "eV/Angstrom. The residual forces of this supercell are subtracted from the "
        "snapshot forces before the fit.",
    )
    rms_displacement: float | None = Field(
        None,
        description="Root mean square displacement of the snapshots from the "
        "reference structure, with the displacement of the center of mass of each "
        "snapshot removed, in Angstrom.",
    )
    rms_displacement_from_force_constants: float | None = Field(
        None,
        description="Classical root mean square displacement at the temperature from "
        "the fitted force constants, in Angstrom. It sums the modes of the "
        "supercell at the Gamma point with a frequency above tol_imaginary_modes.",
    )
    trajectory_health: TrajectoryHealth | None = Field(
        None, description="Check of the MD trajectory."
    )
    tol_imaginary_modes: float | None = Field(
        None, description="Frequencies below -tol_imaginary_modes in THz are imaginary."
    )
    has_imaginary_modes: bool | None = Field(
        None, description="True if the band structure has imaginary modes."
    )
    n_imaginary_modes: int | None = Field(
        None,
        description="Number of imaginary modes at the q-points commensurate with the "
        "supercell.",
    )
    lowest_frequency: float | None = Field(
        None,
        description="Lowest frequency at the q-points commensurate with the "
        "supercell, in THz. Imaginary frequencies are negative.",
    )
    phonon_bandstructure: PhononBandStructureSymmLine | None = Field(
        None, description="Phonon band structure of the effective force constants."
    )
    phonon_dos: PhononDos | None = Field(
        None, description="Phonon density of states of the effective force constants."
    )
    force_constants: ForceConstants | None = Field(
        None,
        description="Effective second-order force constants of the supercell in "
        "eV/Angstrom^2.",
    )
    born: list[Matrix3D] | None = Field(
        None,
        description="Symmetrized Born effective charges. They are used for the "
        "non-analytical correction unless they are all zero.",
    )
    epsilon_static: Matrix3D | None = Field(
        None,
        description="Symmetrized high-frequency dielectric tensor, used for the "
        "non-analytical correction together with born.",
    )
    md_uuids: list[str] | None = Field(None, description="UUIDs of the MD jobs.")
    md_job_dirs: list[str] | None = Field(
        None, description="Directories of the MD jobs."
    )
    npt_input_structure: Structure | None = Field(
        None,
        description="Unit cell whose supercell the NPT MD started from. None without "
        "an NPT MD.",
    )
    pressure: float | None = Field(None, description="Pressure of the NPT MD in kbar.")
    npt_time: float | None = Field(None, description="Length of the NPT MD in ps.")
    npt_equilibration_time: float | None = Field(
        None,
        description="Time at the start of the NPT trajectory left out of the "
        "average cell, in ps.",
    )
    npt_trajectory_health: TrajectoryHealth | None = Field(
        None, description="Check of the NPT trajectory."
    )
    npt_uuid: str | None = Field(None, description="UUID of the NPT MD job.")
    npt_job_dir: str | None = Field(None, description="Directory of the NPT MD job.")
    fixed_cell_relax_uuid: str | None = Field(
        None,
        description="UUID of the relaxation of the atoms in the cell from the NPT MD.",
    )
    fixed_cell_relax_job_dir: str | None = Field(
        None,
        description="Directory of the relaxation of the atoms in the cell from the "
        "NPT MD.",
    )
    uuids: PhononUUIDs | None = Field(
        None,
        description="UUIDs of the relaxation, the phonon displacement calculations "
        "and the Born charge calculation. The undisplaced supercell is the last "
        "phonon displacement calculation.",
    )
    jobdirs: PhononJobDirs | None = Field(
        None,
        description="Directories of the relaxation, the phonon displacement "
        "calculations, the Born charge calculation and the fit.",
    )
