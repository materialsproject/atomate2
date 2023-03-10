"""Flows for Lobster computations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker
from pymatgen.core import Structure

from atomate2.lobster.jobs import PureLobsterMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
from atomate2.vasp.jobs.lobster import (
    VaspLobsterMaker,
    delete_lobster_wavecar,
    get_basis_infos,
    get_lobster_jobs,
    update_user_incar_settings_maker,
)
from atomate2.vasp.sets.core import StaticSetGenerator

__all__ = ["LobsterMaker"]


@dataclass
class LobsterMaker(Maker):
    """
    Maker to perform a Lobster computation.

    Optional optimization.
    Optional static computation with symmetry
    to preconverge the wavefunction.
    Static computation with ISYM=0 is performed
    Several Lobster computations testing several basis sets are performed.
    The basis sets can only be changed with yaml files.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    bulk_relax_maker : .BaseVaspMaker or None
        A maker to perform a relaxation on the bulk.
        Set to ``None`` to skip the
        bulk relaxation
    additional_static_run_maker: .BaseVaspMaker or None
        A maker to perform a preconvergence run
        before the wavefunction computation without symmetry
    vasp_lobster_maker : .BaseVaspMaker
        A maker to perform the computation of the wavefunction before the static run.
        Cannot be skipped.
    delete_all_wavecars:
        if true, all WAVECARs will be deleated after the run
    address_min_basis:
        yaml file including basis set information
    address_max_basis:
        yaml file including basis set information
    """

    name: str = "lobster"
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(RelaxMaker())
    )
    vasp_lobster_maker: BaseVaspMaker = field(
        default_factory=lambda: VaspLobsterMaker()
    )
    additional_static_run_maker: BaseVaspMaker | None = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=StaticSetGenerator(
                user_incar_settings={"LWAVE": True},
                user_kpoints_settings={"grid_density": 6000},
            ),
        )
    )
    lobstermaker: BaseVaspMaker | None = PureLobsterMaker()
    delete_all_wavecars: bool = True
    address_min_basis: str | None = None
    address_max_basis: str | None = None

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

        # do a static WAVECAR computation with symmetry
        # and standard number of bands first
        # this preconverges the WAVECAR
        if self.additional_static_run_maker is not None:
            preconvergence_job = self.additional_static_run_maker.make(
                structure=structure
            )
            preconvergence_job.append_name(" preconvergence")
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

        # Information about the basis is collected
        basis_infos = get_basis_infos(
            structure=structure,
            vaspmaker=self.vasp_lobster_maker,
            address_min_basis=self.address_min_basis,
            address_max_basis=self.address_max_basis,
        )
        jobs.append(basis_infos)

        # Maker needs to be updated here. If the job itself is updated,
        # no further updates on the job are possible
        vaspjob = update_user_incar_settings_maker(
            self.vasp_lobster_maker,
            basis_infos.output["nbands"],
            structure,
            prev_vasp_dir,
        )

        jobs.append(vaspjob)

        static_run_job_dir = vaspjob.output.dir_name
        static_run_uuid = vaspjob.output.uuid

        lobsterjobs = get_lobster_jobs(
            lobstermaker=self.lobstermaker,
            basis_dict=basis_infos.output["basis_dict"],
            wavefunction_dir=vaspjob.output.dir_name,
            optimization_run_job_dir=optimization_run_job_dir,
            optimization_run_uuid=optimization_run_uuid,
            static_run_job_dir=static_run_job_dir,
            static_run_uuid=static_run_uuid,
            additional_static_run_job_dir=additional_static_run_job_dir,
            additional_static_run_uuid=additional_static_run_uuid,
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
                dirs=lobsterjobs.output["lobster_dirs"],
                dir_vasp=vasp_stat,
                dir_preconverge=vasp_add_stat,
            )

            jobs.append(delete_wavecars)

        flow = Flow(jobs, output=lobsterjobs.output)
        return flow
