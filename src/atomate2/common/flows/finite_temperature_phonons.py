"""Flow for effective harmonic phonons at a finite temperature."""

from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
from jobflow import Flow, Maker, OutputReference
from pymatgen.util.due import Doi, due

from atomate2 import SETTINGS
from atomate2.common.jobs.finite_temperature_phonons import (
    fit_finite_temperature_phonons,
    get_md_restart_structure,
    get_md_supercell,
    select_md_snapshots,
)
from atomate2.common.jobs.pheasy import get_supercell_size
from atomate2.common.jobs.phonons import run_phonon_displacements

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from jobflow import Job
    from pymatgen.core import Structure

SUPPORTED_CODES = frozenset(("vasp", "forcefields"))
_THERMOSTATS = ("nose-hoover", "langevin")
_ROTATIONAL_SUM_RULES = ("BH", "H", "BHH")


def _get_force_field(maker: Maker, code: str) -> tuple[str | None, dict | None]:
    """Get the calculator name and keyword arguments of a force field maker."""
    if code != "forcefields":
        return None, None
    name = maker.calculator_meta
    return str(name.value if isinstance(name, Enum) else name), dict(
        maker.calculator_kwargs
    )


@due.dcite(
    Doi("10.48550/arXiv.2508.01020"),
    description="Pheasy code for (an)harmonic force constants.",
)
@due.dcite(
    Doi("10.1103/PhysRevB.84.180301"),
    description="Effective harmonic force constants fitted to MD, as in TDEP.",
)
@dataclass
class BaseFiniteTemperaturePhononMaker(Maker, ABC):
    """
    Maker for effective harmonic phonons at a finite temperature.

    The structure is relaxed first. An NVT MD run at the temperature then starts
    from the undisplaced supercell. The frames of the first equilibration_time
    are left out, and n_snapshots snapshots are picked evenly spread over the
    rest of the trajectory. The phonon displacement maker computes the forces on
    these snapshots and on the undisplaced supercell. The displacements of the
    snapshots are measured from the relaxed positions. pheasy fits one set of
    second-order force constants to the displacements and forces with LASSO, as
    in the temperature-dependent effective potential method. These force
    constants contain the anharmonic effects at the temperature and at the
    volume of the relaxed structure. Thermal expansion is not included. They
    give the phonon band structure and density of states. Imaginary modes are
    reported, not removed.

    The MD maker and the phonon displacement maker are independent. Either can
    be a VASP or a force field maker, so the forces can come from a different
    level of theory than the trajectory. The residual forces of the undisplaced
    supercell always come from the phonon displacement maker.

    The trajectory is also checked. The check looks for melting and for a move
    away from the reference structure. It also looks at the drift of the
    potential energy late in the run. The result is stored in the output
    document. A warning is raised if the check fails, but the fit is still done.

    .. Note::
        The atoms of the relaxed structure are sorted by electronegativity
        before the supercell is built, as the VASP input sets do. The atoms are
        then in the same order in every calculation. The magnetic moments of the
        relaxed structure, if any, are carried over to the supercell, the
        snapshots and every MD job. phonopy and pheasy find the symmetry
        without the magnetic moments, as in the atomate2 phonon workflows. The
        band structure uses the seekpath k-path.

    Parameters
    ----------
    name: str
        Name of the flows produced by this maker.
    temperature: float
        MD temperature in K.
    md_time: float
        Length of the MD trajectory in ps.
    md_time_step: float
        MD time step in fs.
    equilibration_time: float
        Time at the start of the trajectory that is left out, in ps.
    n_snapshots: int
        Number of MD snapshots in the fit.
    thermostat: Literal["nose-hoover", "langevin"]
        Thermostat of the NVT MD.
    md_runs: int
        Number of consecutive MD jobs that make up the trajectory. Each job
        continues from the positions and velocities of the previous one. The
        thermostat variables start again from zero in each job.
    min_length: float
        Each lattice vector of the diagonal supercell is at least min_length
        long, in Angstrom.
    symprec: float
        Symmetry precision for phonopy and pheasy.
    rotational_sum_rule: Literal["BH", "H", "BHH"] | None
        Rotational sum rule passed to pheasy with --rasr. None imposes none.
    alpha_min: int
        Base-10 exponent of the smallest LASSO penalty in the cross-validation,
        passed to pheasy with --alpha_min. The default of -6 is pheasy's own
        default. A warning is raised if the chosen penalty is on either bound of
        the search.
    random_seed: int | None
        Seed of the LASSO fit, and of the initial velocities of a force field
        MD.
    tol_imaginary_modes: float
        Frequencies below -tol_imaginary_modes in THz are imaginary.
    store_force_constants: bool
        Whether to store the force constants in the output document.
    code: str
        Code of the phonon displacement calculations, 'vasp' or 'forcefields'.
    md_code: str
        Code of the MD, 'vasp' or 'forcefields'.
    socket: bool
        If True, the phonon displacement calculations run as a single batched
        calculation. This is not supported by VASP.
    bulk_relax_maker: Maker | None
        Maker for the relaxation of the unit cell. None skips the relaxation.
    born_maker: Maker | None
        VASP maker for the Born effective charges and the dielectric tensor,
        used for the non-analytical correction. None skips it.
    md_maker: Maker
        Maker for the MD. The flow sets its temperature, time step, number of
        steps and thermostat.
    phonon_displacement_maker: Maker
        Maker for the static calculations on the snapshots and the undisplaced
        supercell.
    """

    name: str = "finite temperature phonon"
    temperature: float = 300.0
    md_time: float = 8.0
    md_time_step: float = 1.0
    equilibration_time: float = 1.0
    n_snapshots: int = 50
    thermostat: Literal["nose-hoover", "langevin"] = "nose-hoover"
    md_runs: int = 1
    min_length: float = 12.0
    symprec: float = SETTINGS.PHONON_SYMPREC
    rotational_sum_rule: Literal["BH", "H", "BHH"] | None = "BHH"
    alpha_min: int = -6
    random_seed: int | None = 103
    tol_imaginary_modes: float = 0.1
    store_force_constants: bool = True
    code: str | None = None
    md_code: str | None = None
    socket: bool = False
    bulk_relax_maker: Maker | None = None
    born_maker: Maker | None = None
    md_maker: Maker | None = None
    phonon_displacement_maker: Maker | None = None

    def __post_init__(self) -> None:
        """Check the settings before any calculation is run."""
        if self.md_maker is None or self.phonon_displacement_maker is None:
            raise ValueError("md_maker and phonon_displacement_maker must be set.")
        for key in ("code", "md_code"):
            if getattr(self, key) not in SUPPORTED_CODES:
                raise ValueError(
                    f"{key} must be one of {sorted(SUPPORTED_CODES)}, not "
                    f"{getattr(self, key)!r}."
                )
        if self.thermostat not in _THERMOSTATS:
            raise ValueError(
                f"thermostat must be one of {_THERMOSTATS}, not {self.thermostat!r}."
            )
        if self.rotational_sum_rule not in (*_ROTATIONAL_SUM_RULES, None):
            raise ValueError(
                f"rotational_sum_rule must be one of {_ROTATIONAL_SUM_RULES} or "
                f"None, not {self.rotational_sum_rule!r}."
            )
        for key in ("md_runs", "n_snapshots"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < 1
            ):
                raise ValueError(f"{key} must be a positive integer, not {value!r}.")
        if (
            isinstance(self.alpha_min, bool)
            or not isinstance(self.alpha_min, numbers.Integral)
            or self.alpha_min >= -2
        ):
            raise ValueError(
                f"alpha_min must be an integer below -2, not {self.alpha_min!r}."
            )
        if min(self.temperature, self.md_time, self.md_time_step) <= 0:
            raise ValueError("temperature, md_time and md_time_step must be positive.")
        n_steps = round(self.md_time * 1000 / self.md_time_step)
        n_equil = round(self.equilibration_time * 1000 / self.md_time_step)
        if self.equilibration_time < 0 or n_steps - n_equil < self.n_snapshots:
            raise ValueError(
                f"The MD has {n_steps} steps. After leaving out equilibration_time, "
                f"fewer than the {self.n_snapshots} snapshots are left."
            )
        if self.md_runs > n_steps:
            raise ValueError(
                f"md_runs ({self.md_runs}) is larger than the {n_steps} MD steps."
            )

    def get_md_steps(self) -> list[int]:
        """Get the number of MD steps of each MD job."""
        n_steps = round(self.md_time * 1000 / self.md_time_step)
        base, extra = divmod(n_steps, self.md_runs)
        return [base + (idx < extra) for idx in range(self.md_runs)]

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        supercell_matrix: Matrix3D | None = None,
        born: list[Matrix3D] | None = None,
        epsilon_static: Matrix3D | None = None,
    ) -> Flow:
        """
        Make a flow to calculate effective harmonic phonons at a temperature.

        Parameters
        ----------
        structure: Structure
            The unit cell.
        prev_dir: str | Path | None
            A previous calculation directory. It is passed to the relaxation.
            The born job, the MD jobs and the phonon displacement calculations
            get the relaxation directory, or this directory if there is no
            relaxation.
        supercell_matrix: Matrix3D | None
            Diagonal supercell matrix. If None, it is chosen from min_length.
        born: list[Matrix3D] | None
            Born effective charges of the relaxed structure with its atoms
            sorted by electronegativity. If given with epsilon_static, the
            born_maker is not run.
        epsilon_static: Matrix3D | None
            High-frequency dielectric tensor.

        Returns
        -------
        Flow
        """
        if supercell_matrix is not None and not isinstance(
            supercell_matrix, OutputReference
        ):
            matrix = np.array(supercell_matrix)
            if not np.allclose(matrix, np.diag(np.diag(matrix))):
                raise ValueError("pheasy needs a diagonal supercell matrix.")

        jobs: list[Job | Flow] = []
        optimization_run_job_dir = optimization_run_uuid = None
        born_run_job_dir = born_run_uuid = None

        if self.bulk_relax_maker is not None:
            relax = self.bulk_relax_maker.make(structure, prev_dir=prev_dir)
            jobs.append(relax)
            structure = relax.output.structure
            prev_dir = relax.output.dir_name
            optimization_run_job_dir = relax.output.dir_name
            optimization_run_uuid = relax.output.uuid

        if supercell_matrix is None:
            # pheasy needs a diagonal supercell matrix. With force_diagonal,
            # the size only depends on min_length.
            supercell_job = get_supercell_size(
                structure,
                self.min_length,
                None,
                force_90_degrees=True,
                force_diagonal=True,
            )
            jobs.append(supercell_job)
            supercell_matrix = supercell_job.output

        if self.born_maker is not None and (born is None or epsilon_static is None):
            born_job = self.born_maker.make(structure, prev_dir=prev_dir)
            jobs.append(born_job)
            born = born_job.output.calcs_reversed[0].output.outcar["born"]
            epsilon_static = born_job.output.calcs_reversed[0].output.epsilon_static
            born_run_job_dir = born_job.output.dir_name
            born_run_uuid = born_job.output.uuid

        reference_job = get_md_supercell(
            structure, supercell_matrix, self.symprec, self.code
        )
        jobs.append(reference_job)
        reference = reference_job.output

        all_md_jobs, md_jobs = self.make_md(reference, prev_dir)
        jobs.extend(all_md_jobs)
        md_dirs = [md_job.output.dir_name for md_job in md_jobs]

        snapshot_job = select_md_snapshots(
            md_dirs,
            self.md_code,
            reference,
            self.temperature,
            self.md_time_step,
            self.equilibration_time,
            self.n_snapshots,
        )
        jobs.append(snapshot_job)

        displacement_calcs = run_phonon_displacements(
            displacements=snapshot_job.output["structures"],
            structure=structure,
            supercell_matrix=supercell_matrix,
            phonon_maker=self.phonon_displacement_maker,
            socket=self.socket,
            prev_dir_argname=self.prev_calc_dir_argname,
            prev_dir=prev_dir,
            store_displaced_structures=True,
        )
        jobs.append(displacement_calcs)

        force_field_name, force_field_kwargs = _get_force_field(
            self.phonon_displacement_maker, self.code
        )
        md_force_field_name, md_force_field_kwargs = _get_force_field(
            self.md_maker, self.md_code
        )
        fit_job = fit_finite_temperature_phonons(
            structure=structure,
            supercell_matrix=supercell_matrix,
            snapshot_data=snapshot_job.output,
            displacement_data=displacement_calcs.output,
            temperature=self.temperature,
            thermostat=self.thermostat,
            md_time_step=self.md_time_step,
            equilibration_time=self.equilibration_time,
            code=self.code,
            md_code=self.md_code,
            symprec=self.symprec,
            force_field_name=force_field_name,
            force_field_kwargs=force_field_kwargs,
            md_force_field_name=md_force_field_name,
            md_force_field_kwargs=md_force_field_kwargs,
            md_uuids=[md_job.uuid for md_job in md_jobs],
            md_dirs=md_dirs,
            optimization_run_uuid=optimization_run_uuid,
            optimization_run_job_dir=optimization_run_job_dir,
            born_run_uuid=born_run_uuid,
            born_run_job_dir=born_run_job_dir,
            rotational_sum_rule=self.rotational_sum_rule,
            alpha_min=self.alpha_min,
            random_seed=self.random_seed,
            tol_imaginary_modes=self.tol_imaginary_modes,
            born=born,
            epsilon_static=epsilon_static,
            store_force_constants=self.store_force_constants,
        )
        jobs.append(fit_job)
        return Flow(jobs, fit_job.output, name=self.name)

    def make_md(
        self,
        reference: Structure | OutputReference,
        prev_dir: str | Path | None,
    ) -> tuple[list[Job | Flow], list[Job]]:
        """
        Make the MD jobs, starting from the undisplaced supercell.

        The first MD job starts from the undisplaced supercell. Each later one
        continues from the final positions and velocities of the previous one.

        Parameters
        ----------
        reference: Structure | OutputReference
            The undisplaced supercell.
        prev_dir: str | Path | None
            A previous calculation directory, passed to each MD job.

        Returns
        -------
        tuple[list[Job | Flow], list[Job]]
            All jobs to add to the flow, and the MD jobs in the order of the
            trajectory.
        """
        jobs: list[Job | Flow] = []
        md_jobs: list[Job] = []
        structure = reference
        md_steps = self.get_md_steps()
        for idx, n_steps in enumerate(md_steps, start=1):
            md_job = self.get_md_maker(n_steps).make(structure, prev_dir=prev_dir)
            if len(md_steps) > 1:
                md_job.append_name(f" {idx}/{len(md_steps)}")
            jobs.append(md_job)
            md_jobs.append(md_job)
            if idx < len(md_steps):
                restart = get_md_restart_structure(
                    md_job.output.dir_name, self.md_code, reference
                )
                jobs.append(restart)
                structure = restart.output
        return jobs, md_jobs

    @property
    @abstractmethod
    def prev_calc_dir_argname(self) -> str | None:
        """Name of the argument that passes prev_dir to the phonon displacement maker.

        As this differs between codes, it is implemented by the inheriting class.
        """

    @abstractmethod
    def get_md_maker(self, n_steps: int) -> Maker:
        """
        Get the maker of one MD job, with the settings of this flow.

        Parameters
        ----------
        n_steps: int
            Number of MD steps of the job.

        Returns
        -------
        Maker
        """
