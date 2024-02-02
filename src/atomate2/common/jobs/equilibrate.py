"""Flows adapted from MPMorph *link to origin github repo*"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jobflow import job, Maker, Flow, Response

from emmet.core.tasks import TaskDoc

from atomate2.forcefields.schemas import ForceFieldTaskDocument

import numpy as np
from scipy.optimize import leastsq

if TYPE_CHECKING:

    from pymatgen.core import Structure

MAX_MD_JOBS = (
    9  # if you can't converge with 6 additional calcs you're doing something wrong...
)
BUFFER = 0.1  # gives it enough room to slosh back


@dataclass
class EquilibrateVolumeMaker(Maker):
    name: str = "equilibrate volume seacher"
    scale_factor_increment: float = 0.2
    convergence_md_maker: Maker | None = Maker

    @job
    def make(
        self,
        structure: Structure,
        working_task_docs: (
            list[TaskDoc] | list[ForceFieldTaskDocument]
        ) = None,  # accounts for forcefield task docs, is there a more general way to do this?
    ):
        """
        Generate scaled unit cell structures, then finds equilibrium volume structure.

        An equation of state (Birch-Murnaghan) is fit to the pressure-volume data from the scaled structures.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        working_task_docs : list[TaskDoc] or list[ForceFieldTaskDocument] or None
            None if this is the first time the job is being run, otherwise a list of task documnets
            corresponding to each of the scaled structures that have been run so far.

        Returns
        -------
        Structure
            Equilibrated structure from equation of state.

        Response
        _______
        Flow
            If working_task_docs is None or equation of state fitting does is insufficient.
            A list of scaled structures molecular dynamics jobs.
        """
        if working_task_docs is not None and len(working_task_docs) > MAX_MD_JOBS:
            raise RuntimeError(
                "Maximum number of MD runs for equilibrium volume search exceeded"
            )

        if working_task_docs is None:
            initial_scale_factors = [
                1 - self.scale_factor_increment,
                1,
                1 + self.scale_factor_increment,
            ]

            scaled_structs = [
                get_scaled_structure(structure, factor)
                for factor in initial_scale_factors
            ]

            new_jobs = [
                self.convergence_md_maker.make(struct) for struct in scaled_structs
            ]

            working_task_docs = [job.output for job in new_jobs]

        else:
            volumes = [doc.output.structure.volume for doc in working_task_docs]
            pressures = [
                1 / 3 * np.trace(doc.output.stress) for doc in working_task_docs
            ]

            pv_pairs = np.array(list(zip(pressures, volumes)))

            max_explored_volume = max(volumes)
            min_explored_volume = min(volumes)

            new_job_vol_scales = []
            try:
                params = fit_BirchMurnaghanPV_EOS(pv_pairs)
                equil_volume = params[0]
                if (
                    equil_volume < max_explored_volume
                    and equil_volume > min_explored_volume
                ):
                    final_structure = structure.copy()
                    final_structure.scale_lattice(equil_volume)
                    return final_structure
                elif equil_volume > max_explored_volume:
                    new_job_vol_scales.append(
                        get_new_max_volume(equil_volume, structure)
                    )
                elif equil_volume < min_explored_volume:
                    new_job_vol_scales.append(
                        get_new_min_volume(equil_volume, structure)
                    )
            except ValueError:
                print(
                    "Unable to converge EoS fit for volume optimization, expanding search range"
                )
                new_job_max = expand_upper_bound(max_explored_volume, structure)
                new_job_min = expand_lower_bound(max_explored_volume, structure)
                new_job_vol_scales.append(new_job_max)
                new_job_vol_scales.append(new_job_min)

            # This is specific to the type of MD run you're doing
            scaled_structs = [
                get_scaled_structure(structure, factor) for factor in new_job_vol_scales
            ]

            new_jobs = [
                self.convergence_md_maker.make(struct) for struct in scaled_structs
            ]

            for new_job in new_jobs:
                working_task_docs.append(new_job.output)

        expanded_search_job = self.make(structure, working_task_docs)

        flow = Flow([*new_jobs, expanded_search_job])

        return Response(replace=flow, output=expanded_search_job.output)


# Do these guys belong here or somewhere else?


def get_scaled_structure(struct: Structure, scale_factor: float):
    copy: Structure = struct.copy()
    copy.scale_lattice(copy.volume * scale_factor)
    return copy


def get_new_max_volume(equil_guess, original_structure):
    return equil_guess / original_structure.volume + BUFFER


def expand_upper_bound(old_max_vol, original_structure):
    return old_max_vol / original_structure.volume + 0.2


def get_new_min_volume(equil_guess, original_structure):
    return equil_guess / original_structure.volume - BUFFER


def expand_lower_bound(old_max_vol, original_structure):
    return old_max_vol / original_structure.volume - 0.2


def BirchMurnaghanPV_EOS(V, params):
    """
    Args:
        V: volume
        params: tuple of B0,V0,B0p
    Returns:
        Pressure of Birch-Murnaghan EOS at V with given parameters E0, B0, V0 and B0p
    """
    V0, B0, B0p = params[0], params[1], params[2]
    n = (V0 / V) ** (1.0 / 3)  # Note this definition is different from the Energy EOS
    p = 3.0 / 2.0 * B0 * (n**7 - n**5) * (1.0 + 3.0 / 4 * (B0p - 4) * (n**2 - 1.0))
    return p


def fit_BirchMurnaghanPV_EOS(p_v: np.ndarray):
    # Borrows somewhat from pymatgen/io/abinitio/EOS
    # Initial guesses for the parameters
    eqs = np.polyfit(p_v[:, 1], p_v[:, 0], 2)
    V0 = np.mean(p_v[:, 1])  # still use mean to ensure we are at reasonable volumes
    B0 = -1 * (2 * eqs[0] * V0**2 + eqs[1] * V0)
    B0p = 4.0
    initial_params = (V0, B0, B0p)
    Error = lambda params, x, y: BirchMurnaghanPV_EOS(x, params) - y
    found_params, check = leastsq(Error, initial_params, args=(p_v[:, 1], p_v[:, 0]))
    if check not in [1, 2, 3, 4]:
        raise ValueError("fitting not converged")
    else:
        return found_params
