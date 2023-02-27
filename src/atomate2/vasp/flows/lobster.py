"""Flows for Lobster computations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import StaticMaker, RelaxMaker, TightRelaxMaker
from atomate2.vasp.jobs.lobster import (
    VaspLobsterMaker,
    get_basis_infos,
    get_lobster_jobs,
    update_user_incar_settings_job,
    delete_lobster_wavecar,
)
from atomate2.vasp.sets.core import StaticSetGenerator
from jobflow import Flow
from jobflow import Maker
from pymatgen.core import Structure

__all__ = ["LobsterMaker"]


@dataclass
class LobsterMaker(Maker):
    """
    Maker to perform a Lobster computation

    Optional optimization.
    Optional static computation with symmetry
    static computation with ISYM=0 is performed
    several Lobster computations testing several basis sets is performed

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
        bulk_relax_maker : .BaseVaspMaker or None
        A maker to perform a tight relaxation on the bulk.
        Set to ``None`` to skip the
        bulk relaxation
    static_energy_maker : .BaseVaspMaker or None
        A maker to perform the computation of the DFT energy on the bulk.
        Set to ``None`` to skip the
        static energy computation
    code: str
        determines the dft code. currently only vasp is implemented.
        This keyword might enable the implementation of other codes
        in the future
    """

    name: str = "lobster"
    # implement different calculation types
    calculation_type: str = "standard",
    delete_all_wavecars: bool = True,
    user_lobsterin_settings: dict | None = None
    user_incar_settings: dict | None = None
    user_kpoints_settings: dict | None = None
    user_supplied_basis: dict | None = None
    isym: int = 0,
    additional_outputs: list[str] | None = None,
    additional_optimization: bool = False,
    additional_static_run: bool = True  # will add an additional static run
    user_incar_settings_optimization: dict | None = None,
    user_kpoints_settings_optimization: dict | None = None,
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(RelaxMaker())
    )
    vasp_lobster_maker: BaseVaspMaker = field(
        default_factory=lambda: VaspLobsterMaker()
    )
    additional_static_run_maker: BaseVaspMaker | None = field(
        default_factory=lambda: StaticMaker(
            name="preconvergence run",
            input_set_generator=StaticSetGenerator(
                user_incar_settings={"LWAVE": True, "ISMEAR": 0},
                user_kpoints_settings={"grid_density": 100},
            ),
        )
    )

    def make(
            self,
            structure: Structure,
            prev_vasp_dir: str | Path | None = None,
    ):
        """
        Make flow to calculate bonding properties.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure. Please start with a structure
            that is nearly fully optimized as the internal optimizers
            have very strict settings!
        prev_vasp_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.

        """

        jobs = []

        # do a relaxation step first
        if self.bulk_relax_maker is not None:
            # optionally relax the structure
            bulk = self.bulk_relax_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
            jobs.append(bulk)
            structure = bulk.output.structure
            optimization_run_job_dir = bulk.output.dir_name
            optimization_run_uuid = bulk.output.uuid
        else:
            optimization_run_job_dir = None
            optimization_run_uuid = None

        # do a static WAVECAR computation with symmetry and standard number of bands first
        # Do a static VASP computation
        if self.additional_static_run_maker is not None:
            preconvergence_job = self.additional_static_run_maker.make(
                structure=structure
            )
            jobs.append(preconvergence_job)
            prev_vasp_dir = preconvergence_job.output.dir_name
            additional_static_run_job_dir = preconvergence_job.output.dir_name
            additional_static_run_uuid = preconvergence_job.output.uuid
        else:
            if optimization_run_job_dir is not None:
                prev_vasp_dir = optimization_run_job_dir
            else:
                prev_vasp_dir = None
            additional_static_run_job_dir = None
            additional_static_run_uuid = None

        # at gamma: -5 is used as standard, leads to errors for gamma only
        vaspjob = self.vasp_lobster_maker.make(
            structure=structure, prev_vasp_dir=prev_vasp_dir
        )

        basis_infos = get_basis_infos(
            structure=structure,
            vaspmaker=self.vasp_lobster_maker,
            address_min_basis=None,
            address_max_basis=None,
        )
        jobs.append(basis_infos)
        # nbands=basis_infos.output["nbands"]
        # vaspjob = update_user_incar_settings(vaspjob,incar_updates={"NBANDS":8}, name_filter='static_run')
        vaspjob = update_user_incar_settings_job(vaspjob, basis_infos.output)

        jobs.append(vaspjob)

        static_run_job_dir=vaspjob.output.dir_name
        static_run_uuid=vaspjob.output.uuid

        lobsterjobs = get_lobster_jobs(
            basis_infos.output["basis_dict"], vaspjob.output.dir_name,
            user_lobsterin_settings=self.user_lobsterin_settings, additional_outputs=self.additional_outputs
        )

        jobs.append(lobsterjobs)


        # will delete all WAVECARs that have been copied

        if self.delete_all_wavecars:
            vasp_stat = vaspjob.output.dir_name
            if self.additional_static_run_maker is not None:
                vasp_add_stat = preconvergence_job.output.dir_name
            if self.additional_static_run_maker is None:
                vasp_add_stat = None

            delete_wavecars = delete_lobster_wavecar(
                dirs=lobsterjobs.output["dirs"],
                dir_vasp=vasp_stat,
                dir_preconverge=vasp_add_stat,
            )

            jobs.append(delete_wavecars)
        outputs={}
        outputs["optimization_run_job_dir"]=optimization_run_job_dir
        outputs["optimization_run_uuid"]=optimization_run_uuid
        outputs["static_run_job_dir"]=static_run_job_dir
        outputs["static_run_uuid"]=static_run_uuid
        outputs["additional_static_run_dir"]=additional_static_run_job_dir
        outputs["additional_static_uuid"]=additional_static_run_uuid
        outputs["lobster_job_dirs"]=lobsterjobs.output["dirs"]
        outputs["lobster_uuids"]=lobsterjobs.output["uuids"]
        outputs["lobster_task_documents"]=lobsterjobs.output["lobster_task_documents"]
        flow = Flow(jobs, output=outputs)
        return flow
