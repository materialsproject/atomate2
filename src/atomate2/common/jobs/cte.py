"""Jobs for thermal expansion from mode Grueneisen tensors."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import job

from atomate2.common.jobs.gruneisen import PhononDoc, _get_taskdoc_run_dir
from atomate2.common.jobs.pheasy import _ANHARMONIC_FIT_METHODS, _DEFAULT_FILE_PATHS
from atomate2.common.schemas.cte import CTEDocument

if TYPE_CHECKING:
    from collections.abc import Sequence

    from emmet.core.math import MatrixVoigt
    from pymatgen.core import Structure


@job(output_schema=CTEDocument)
def compute_cte(
    phonon_output: PhononDoc,
    elastic_tensor: MatrixVoigt,
    elastic_structure: Structure,
    anhar_fit_methods: Sequence[str] = ("one-shot",),
    temperatures: Sequence[float] = tuple(range(0, 1001, 10)),
    mesh: tuple[int, int, int] | float = (12, 12, 12),
    tol_imaginary_modes: float = 0.1,
    min_frequency: float = 1e-3,
    symprec: float = 1e-5,
) -> CTEDocument:
    """
    Compute the thermal expansion from the pheasy force constants.

    The second- and third-order force constants of each fit method are read
    from the folder of the pheasy fit, so this job must run where that folder
    can be read. The cocktail fit uses the second-order force constants of the
    harmonic fit. The one-shot fit uses its own.

    Parameters
    ----------
    phonon_output: PhononDoc
        Output document of the pheasy phonon flow, run with cal_anhar_fcs=True.
    elastic_tensor: MatrixVoigt
        Elastic tensor in GPa and Voigt notation, as ElasticDocument's
        elastic_tensor.raw.
    elastic_structure: Structure
        Structure of the elastic calculation. Its lattice must match the unit
        cell of the phonon run, so that both tensors are in the same frame.
    anhar_fit_methods: Sequence[str]
        Fit methods whose force constants are used, "cocktail" and/or "one-shot".
    temperatures: Sequence[float]
        Temperatures in K, not negative.
    mesh: tuple[int, int, int] | float
        q-point mesh, or a q-point density used as kppa in pymatgen's
        Kpoints.automatic_density for the unit cell.
    tol_imaginary_modes: float
        If a frequency on the mesh is below -tol_imaginary_modes in THz, a
        warning is raised and the thermal expansion of that fit is not computed.
    min_frequency: float
        Modes below this frequency in THz are left out.
    symprec: float
        Symmetry precision passed to phono3py.

    Returns
    -------
    CTEDocument
    """
    unknown = set(anhar_fit_methods) - set(_ANHARMONIC_FIT_METHODS)
    if unknown or not anhar_fit_methods:
        raise ValueError(
            f"anhar_fit_methods must be a non-empty subset of "
            f"{_ANHARMONIC_FIT_METHODS}, not {list(anhar_fit_methods)}."
        )
    job_dir_name = _get_taskdoc_run_dir(phonon_output)
    if job_dir_name is None:
        raise ValueError("The phonon output does not record the pheasy job folder.")
    job_dir = Path(job_dir_name)

    # the files written by the pheasy fits
    one_shot_dir = job_dir / _DEFAULT_FILE_PATHS["one_shot_dir"]
    files = {
        "cocktail": (
            job_dir / _DEFAULT_FILE_PATHS["force_constants"],
            job_dir / "fc3.hdf5",
        ),
        "one-shot": (one_shot_dir / "fc2.hdf5", one_shot_dir / "fc3.hdf5"),
    }
    force_constant_files = {method: files[method] for method in anhar_fit_methods}

    return CTEDocument.from_force_constants(
        phonopy_yaml=job_dir / _DEFAULT_FILE_PATHS["phonopy"],
        force_constant_files=force_constant_files,
        elastic_tensor=elastic_tensor,
        structure=elastic_structure,
        temperatures=temperatures,
        mesh=mesh,
        tol_imaginary_modes=tol_imaginary_modes,
        min_frequency=min_frequency,
        symprec=symprec,
    )
