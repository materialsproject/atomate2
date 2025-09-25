"""Base DFPT abinit flow makers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from abipy.abio.factories import scf_for_phonons
from jobflow import Flow, Maker

from atomate2.abinit.jobs.core import StaticMaker
from atomate2.abinit.jobs.mrgddb import MrgddbMaker
from atomate2.abinit.jobs.response import generate_perts
from atomate2.abinit.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure
    from pymatgen.io.abinit.abiobjects import KSampling

    from atomate2.abinit.jobs.base import BaseAbinitMaker


@dataclass
class DfptFlowMaker(Maker):
    """
    Maker to generate a DFPT flow with abinit.

    The classmethods allow to tailor the flow for specific properties
        accessible via DFPT.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    static_maker : .BaseAbinitMaker
        The maker to use for the static calculation.
    ddk_maker : .BaseAbinitMaker
        The maker to use for the DDK calculations.
    dde_maker : .BaseAbinitMaker
        The maker to use for the DDE calculations.
    dte_maker : .BaseAbinitMaker
        The maker to use for the DTE calculations.
    wfq_maker : .BaseAbinitMaker
        The maker to use for the WFQ calculations.
    phonon_maker : .BaseAbinitMaker
        The maker to use for the phonon calculations.
    mrgddb_maker : .Maker
        The maker to merge the DDBs.
    mrgdv_maker : .Maker
        The maker to merge the POT files.
    anaddb_maker : .Maker
        The maker to analyze the DDBs.
    use_dde_sym : bool
        True if only the irreducible DDE perturbations should be considered,
        False otherwise.
    dte_skip_permutations: Since the current version of abinit always performs
        all the permutations of the perturbations, even if only one is asked,
        if True avoids the creation of inputs that will produce duplicated outputs.
    qpt_list: list or tuple or None
        A list of q points to compute the phonons.
    ngqpt: list or tuple or None
        Monkhorst-Pack divisions for the phonon q-mesh.
        Default is the same as the one used in the GS calculation.
    user_qpoints_settings: dict or KSampling or None
        Allows user to define the qmesh by supplying a dict. E.g.,
        ``{"reciprocal_density": 1000}``. User can also supply a KSampling object.
    qptopt: int or None
        Option for the generation of the q-points list, default same as kptopt in gs.
    """

    name: str = "DFPT"
    static_maker: BaseAbinitMaker = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=StaticSetGenerator(factory=scf_for_phonons)
        )
    )
    ddk_maker: BaseAbinitMaker | None = None  # |
    dde_maker: BaseAbinitMaker | None = None  # |
    dte_maker: BaseAbinitMaker | None = None  # |
    wfq_maker: BaseAbinitMaker | None = None  # |
    phonon_maker: BaseAbinitMaker | None = None  # |
    mrgddb_maker: Maker | None = field(default_factory=MrgddbMaker)  # |
    mrgdv_maker: Maker | None = None  # |
    anaddb_maker: Maker | None = None  # |
    use_dde_sym: bool = True
    dte_skip_permutations: bool | None = False
    qpt_list: list[list] | None = None
    ngqpt: list | None = None
    qptopt: int | None = None
    user_qpoints_settings: dict | KSampling | None = None

    def __post_init__(self) -> None:
        """Process post-init configuration."""
        if self.dde_maker and not self.ddk_maker:
            raise ValueError(
                "DDK calculations are required to continue \
                with the DDE calculations. Either provide a DDK Maker \
                or remove the DDE one."
            )
        if self.dte_maker and not self.dde_maker:
            raise ValueError(
                "DDE calculations are required to continue \
                with the DTE calculations. Either provide a DDE Maker \
                or remove the DTE one."
            )
        if self.dte_maker and self.use_dde_sym:
            raise ValueError(
                "DTE calculations require all the DDE perturbations, \
                the use of symmetries is not allowed."
            )
        if self.anaddb_maker and not self.mrgddb_maker:
            raise ValueError(
                "Anaddb should be used to analyze a merged DDB. \
                Either provide a Mrgddb Maker \
                or remove the AnaddbMaker."
            )
        if (
            np.sum(
                [
                    x is not None
                    for x in [self.ngqpt, self.qpt_list, self.user_qpoints_settings]
                ]
            )
            > 1
        ):
            raise ValueError(
                "You can only provide one of ngqpt, qpt_list or user_qpoints_settings."
            )

    def make(
        self,
        structure: Structure | None = None,
        restart_from: str | Path | None = None,
    ) -> Flow:
        """
        Create a DFPT flow.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        restart_from : str or Path or None
            One previous directory to restart from.
        anaddb_kwargs : dict
            Additional kwargs for the anaddb maker.

        Returns
        -------
        Flow
            A DFPT flow
        """
        jobs = []
        static_job = self.static_maker.make(
            structure=structure, restart_from=restart_from
        )
        jobs.append(static_job)

        if self.ddk_maker:
            # the use of symmetries is not implemented for DDK
            perturbations = [{"idir": 1}, {"idir": 2}, {"idir": 3}]
            ddk_jobs = []
            outputs: dict[str, list] = {"dirs": []}
            for ipert, pert in enumerate(perturbations):
                ddk_job = self.ddk_maker.make(
                    perturbation=pert,
                    prev_outputs=static_job.output.dir_name,
                )
                ddk_job.append_name(f"{ipert + 1}/{len(perturbations)}")

                ddk_jobs.append(ddk_job)
                outputs["dirs"].append(ddk_job.output.dir_name)

            ddk_calcs = Flow(ddk_jobs, outputs)
            jobs.append(ddk_calcs)

        pert_jobs_generator = generate_perts(
            gsinput=static_job.output.input.abinit_input,
            skip_dte_permutations=self.dte_skip_permutations,
            use_dde_symmetries=self.use_dde_sym,
            ngqpt=self.ngqpt,
            qptopt=self.qptopt,
            qpt_list=self.qpt_list,
            user_qpoints_settings=self.user_qpoints_settings,
            dde_maker=self.dde_maker,
            wfq_maker=self.wfq_maker,
            phonon_maker=self.phonon_maker,
            dte_maker=self.dte_maker,
            scf_output=static_job.output.dir_name,
            ddk_output=None if self.ddk_maker is None else ddk_calcs.output["dirs"],
        )
        jobs.append(pert_jobs_generator)

        if self.mrgddb_maker:
            # merge the DDE, DTE and Phonon DDB.
            prev_outputs = [
                pert_jobs_generator.output["dirs"][key]
                for key, maker in [
                    ("dde", self.dde_maker),
                    ("dte", self.dte_maker),
                    ("phonon", self.phonon_maker),
                ]
                if maker
            ]

            mrgddb_job = self.mrgddb_maker.make(
                prev_outputs=prev_outputs,
            )

            jobs.append(mrgddb_job)

        if self.mrgdv_maker:
            # merge the DDE and Phonon POT files.
            prev_outputs = []
            if self.ddk_maker:
                prev_outputs.extend(ddk_calcs.output["dirs"])
            prev_outputs = prev_outputs + [
                pert_jobs_generator.output["dirs"][key]
                for key, maker in [
                    ("dde", self.dde_maker),
                    ("dte", self.dte_maker),
                    ("phonon", self.phonon_maker),
                ]
                if maker
            ]

            mrgdv_job = self.mrgdv_maker.make(
                prev_outputs=prev_outputs,
            )
            jobs.append(mrgdv_job)

        # It could be possible to handle the case of qpt_list not None by
        # using a different anaddb_maker that calculates only the frequencies
        # at the selected qpoints. Unlikely case. Not handled at the moment.
        if self.anaddb_maker and self.qpt_list is None:
            # analyze a merged DDB.
            anaddb_job = self.anaddb_maker.make(
                structure=static_job.output.structure,
                prev_outputs=mrgddb_job.output.dir_name,
            )

            jobs.append(anaddb_job)

        return Flow(
            jobs, output=[j.output for j in jobs], name=self.name
        )  # TODO: fix outputs
