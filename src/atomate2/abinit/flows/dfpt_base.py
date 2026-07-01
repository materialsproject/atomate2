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

__all__ = ["DfptFlowMaker"]


@dataclass
class DfptFlowMaker(Maker):
    """
    Maker to generate a DFPT (Density Functional Perturbation Theory) flow.

    This maker creates workflows for calculating response properties using DFPT
    with ABINIT. The flow can include various types of perturbations (DDK, DDE,
    DTE, WFQ, phonon) and optional post-processing steps. Classmethods are
    available to tailor the flow for specific properties.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    static_maker : .BaseAbinitMaker
        The maker for the initial static self-consistent calculation.
        Defaults to StaticMaker with scf_for_phonons factory.
    ddk_maker : .BaseAbinitMaker or None
        The maker for DDK (derivative of wavefunctions with respect to k)
        calculations. Required for electric field perturbations. If None,
        DDK calculations are skipped.
    dde_maker : .BaseAbinitMaker or None
        The maker for DDE (derivative with respect to electric field)
        calculations. Requires ddk_maker to be set. If None, DDE
        calculations are skipped.
    dte_maker : .BaseAbinitMaker or None
        The maker for DTE (derivative with respect to strain and electric
        field) calculations. Requires dde_maker to be set. If None, DTE
        calculations are skipped.
    wfq_maker : .BaseAbinitMaker or None
        The maker for WFQ (wavefunctions at q-points) calculations.
        If None, WFQ calculations are skipped.
    phonon_maker : .BaseAbinitMaker or None
        The maker for phonon calculations (atomic displacement
        perturbations). If None, phonon calculations are skipped.
    mrgddb_maker : .Maker or None
        The maker to merge derivative database (DDB) files from different
        perturbations. Defaults to MrgddbMaker.
    mrgdv_maker : .Maker or None
        The maker to merge first-order potential (POT) files. If None,
        POT file merging is skipped.
    anaddb_maker : .Maker or None
        The maker to analyze merged DDB files using ANADDB. Requires
        mrgddb_maker to be set. If None, ANADDB analysis is skipped.
    use_dde_sym : bool
        If True, use symmetries to reduce the number of DDE perturbations
        to only irreducible ones. Cannot be True if dte_maker is set.
        Default is True.
    dte_skip_permutations : bool or None
        If True, skip creation of redundant DTE inputs. Since ABINIT
        always computes all permutations of perturbations even when only
        one is requested, this avoids duplicated outputs. Default is True.
    qpt_list : list[list] or None
        Explicit list of q-points for phonon calculations. Cannot be used
        with ngqpt or user_qpoints_settings. Note: When using an explicit
        q-point list, anaddb post-processing (phonon bands and DOS) is not
        performed. For full phonon analysis, use ngqpt or
        user_qpoints_settings to define a uniform q-point grid instead.
    ngqpt : list or None
        Monkhorst-Pack grid divisions for the phonon q-point mesh (e.g.,
        [4, 4, 4]). If None (and qpt_list and user_qpoints_settings are
        also None), defaults to the same grid as the ground state k-point
        mesh. Cannot be used with qpt_list or user_qpoints_settings.
    user_qpoints_settings : dict or KSampling or None
        Custom q-point mesh settings. Can be a dict (e.g.,
        {"reciprocal_density": 1000}) or a KSampling object. Cannot be
        used with qpt_list or ngqpt.
    qptopt : int or None
        Option for q-point generation. If None, defaults to the same as
        kptopt from the ground state calculation.
    """

    name: str = "DFPT"
    static_maker: BaseAbinitMaker = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=StaticSetGenerator(factory=scf_for_phonons)
        )
    )
    ddk_maker: BaseAbinitMaker | None = None
    dde_maker: BaseAbinitMaker | None = None
    dte_maker: BaseAbinitMaker | None = None
    wfq_maker: BaseAbinitMaker | None = None
    phonon_maker: BaseAbinitMaker | None = None
    mrgddb_maker: Maker | None = field(default_factory=MrgddbMaker)
    mrgdv_maker: Maker | None = None
    anaddb_maker: Maker | None = None
    use_dde_sym: bool = True
    dte_skip_permutations: bool | None = True
    qpt_list: list[list] | None = None
    ngqpt: list | None = None
    qptopt: int | None = None
    user_qpoints_settings: dict | KSampling | None = None

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.

        This method checks for incompatible parameter combinations and
        enforces dependencies between different makers. It ensures:
        - DDK calculations are present when DDE is requested
        - DDE calculations are present when DTE is requested
        - DTE is not used with DDE symmetries
        - ANADDB has a merged DDB to analyze
        - Only one q-point specification method is used

        Raises
        ------
        ValueError
            If invalid parameter combinations are detected.
        """
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

        This method creates a complete DFPT workflow including static calculations,
        perturbation calculations (DDK, DDE, DTE, WFQ, phonon), and optional
        post-processing steps (merging DDB files and running Anaddb analysis).

        Parameters
        ----------
        structure : Structure or None
            A pymatgen Structure object. If None, the structure should be provided
            through the restart_from directory.
        restart_from : str or Path or None
            Path to a previous calculation directory to restart from. This allows
            reusing wavefunctions and density from a completed calculation.

        Returns
        -------
        Flow
            A jobflow Flow containing all the jobs for the DFPT calculation,
            including the static job, perturbation jobs, and optional
            merge/analysis jobs.
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
                ddk_job.append_name(
                    f" - {ipert + 1}/{len(perturbations)}", dynamic=False
                )

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
