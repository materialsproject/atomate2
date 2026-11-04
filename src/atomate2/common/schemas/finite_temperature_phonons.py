"""Schemas for the finite-temperature phonon workflow."""

from pydantic import Field
from pymatgen.core import Structure

from atomate2.common.schemas.phonons import PhononBSDOSDoc


class FiniteTemperaturePhononDoc(PhononBSDOSDoc):
    """Effective harmonic phonons at a finite temperature.

    structure is the unit cell of the NVT MD and the fit, with its atoms sorted
    by electronegativity. It is the relaxed unit cell, or the cell from the NPT
    MD if there was one. code is the code of the phonon displacement
    calculations. born and epsilon_static are the symmetrized values used for
    the non-analytical correction. has_imaginary_modes is True if the band
    structure has imaginary modes. The undisplaced supercell is the last
    phonon displacement calculation in uuids and jobdirs. The thermodynamic
    properties are not computed.
    """

    temperature: float | None = Field(None, description="MD temperature in K.")
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
    tol_imaginary_modes: float | None = Field(
        None, description="Frequencies below -tol_imaginary_modes in THz are imaginary."
    )
    n_imaginary_modes: int | None = Field(
        None,
        description="Number of frequencies below -tol_imaginary_modes, summed over "
        "all q-points commensurate with the supercell. q-points that are equivalent "
        "by symmetry are each counted.",
    )
    lowest_frequency: float | None = Field(
        None,
        description="Lowest frequency at the q-points commensurate with the "
        "supercell, in THz. Imaginary frequencies are negative.",
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
    npt_uuids: list[str] | None = Field(None, description="UUIDs of the NPT MD jobs.")
    npt_job_dirs: list[str] | None = Field(
        None, description="Directories of the NPT MD jobs."
    )
    fixed_cell_relax_uuid: str | None = Field(
        None,
        description="UUID of the relaxation of the atoms in the cell from the NPT MD.",
    )
    fixed_cell_relax_job_dir: str | None = Field(
        None,
        description="Directory of the relaxation of the atoms in the cell from the "
        "NPT MD.",
    )
