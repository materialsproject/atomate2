"""Flows for Lobster computations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker
from monty.dev import requires

from atomate2.lobster.jobs import LobsterMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker, UniformBandStructureMaker
from atomate2.vasp.jobs.core import NonSCFMaker, RelaxMaker, StaticMaker
from atomate2.vasp.jobs.lobster import (
    delete_lobster_wavecar,
    get_basis_infos,
    get_lobster_jobs,
    update_user_incar_settings_maker,
)
from atomate2.vasp.sets.core import NonSCFSetGenerator, StaticSetGenerator

try:
    import ijson
    from lobsterpy.cohp.analyze import Analysis
    from lobsterpy.cohp.describe import Description
except ImportError:
    ijson = None
    Analysis = None
    Description = None

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker


LOBSTER_UNIFORM_MAKER = UniformBandStructureMaker(
    name="uniform lobster structure",
    static_maker=StaticMaker(
        input_set_generator=StaticSetGenerator(
            user_kpoints_settings={"reciprocal_density": 100},
            user_incar_settings={
                "EDIFF": 1e-7,
                "LAECHG": False,
                "LVTOT": False,
                "LREAL": False,
                "ALGO": "Normal",
                "LWAVE": False,
            },
        )
    ),
    bs_maker=NonSCFMaker(
        input_set_generator=NonSCFSetGenerator(
            user_kpoints_settings={"reciprocal_density": 400},
            user_incar_settings={
                "LWAVE": True,
                "ISYM": 0,
            },
        ),
        task_document_kwargs={"parse_dos": False, "parse_bandstructure": False},
    ),
)


@dataclass
class VaspLobsterMaker(Maker):
    """
    Maker to perform a Lobster computation.

    The calculations performed are:

    1. Optional optimization.
    2. Static calculation with ISYM=0.
    3. Several Lobster computations testing several basis sets are performed.

    .. Note::

        The basis sets can only be changed with yaml files.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker : .BaseVaspMaker or None
        A maker to perform a relaxation on the bulk. Set to ``None`` to skip the
        bulk relaxation.
    lobster_static_maker : .BaseVaspMaker
        A maker to perform the computation of the wavefunction before the static run.
        Cannot be skipped. It can be LOBSTERUNIFORM or LobsterStaticMaker()
    lobster_maker : .LobsterMaker
        A maker to perform the Lobster run.
    delete_wavecars : bool
        If true, all WAVECARs will be deleted after the run.
    address_min_basis : str
        A path to a yaml file including basis set information.
    address_max_basis : str
       A path to a yaml file including basis set information.
    """

    name: str = "lobster"
    relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(RelaxMaker())
    )
    lobster_static_maker: BaseVaspMaker = field(
        default_factory=lambda: LOBSTER_UNIFORM_MAKER
    )
    lobster_maker: LobsterMaker | None = field(default_factory=LobsterMaker)
    delete_wavecars: bool = True
    address_min_basis: str | None = None
    address_max_basis: str | None = None

    @requires(
        Analysis,
        "This flow requires lobsterpy and ijson to function properly. "
        "Please reinstall atomate2 using atomate2[lobster]",
    )
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """Make flow to calculate bonding properties.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure. Please start with a structure
            that is nearly fully optimized as the internal optimizers
            have very strict settings!
        prev_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.
        """
        jobs = []

        # optionally relax the structure
        optimization_dir = None
        optimization_uuid = None
        if self.relax_maker is not None:
            optimization = self.relax_maker.make(structure, prev_dir=prev_dir)
            jobs.append(optimization)
            structure = optimization.output.structure
            optimization_dir = optimization.output.dir_name
            optimization_uuid = optimization.output.uuid
            prev_dir = optimization_dir

        # Information about the basis is collected
        basis_infos = get_basis_infos(
            structure=structure,
            vasp_maker=self.lobster_static_maker,
            address_min_basis=self.address_min_basis,
            address_max_basis=self.address_max_basis,
        )
        jobs.append(basis_infos)

        # Maker needs to be updated here. If the job itself is updated, no further
        # updates on the job are possible
        lobster_static = update_user_incar_settings_maker(
            self.lobster_static_maker,
            basis_infos.output["nbands"],
            structure,
            prev_dir,
        )
        jobs.append(lobster_static)
        lobster_static_dir = lobster_static.output.dir_name
        lobster_static_uuid = lobster_static.output.uuid

        lobster_jobs = get_lobster_jobs(
            lobster_maker=self.lobster_maker,
            basis_dict=basis_infos.output["basis_dict"],
            optimization_dir=optimization_dir,
            optimization_uuid=optimization_uuid,
            static_dir=lobster_static_dir,
            static_uuid=lobster_static_uuid,
        )
        jobs.append(lobster_jobs)

        # delete all WAVECARs that have been copied
        # TODO:  this has to be adapted as well
        if self.delete_wavecars:
            delete_wavecars = delete_lobster_wavecar(
                dirs=lobster_jobs.output["lobster_dirs"],
                lobster_static_dir=lobster_static.output.dir_name,
            )
            jobs.append(delete_wavecars)

        return Flow(jobs, output=lobster_jobs.output)
