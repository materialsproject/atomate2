"""DFPT abinit flow makers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from abipy.abio.factories import scf_for_phonons
from jobflow import Flow, Maker

from atomate2.abinit.jobs.anaddb import AnaddbDfptDteMaker, AnaddbMaker
from atomate2.abinit.jobs.core import StaticMaker
from atomate2.abinit.jobs.mrgddb import MrgddbMaker
from atomate2.abinit.jobs.response import (
    DdeMaker,
    DdkMaker,
    DteMaker,
    generate_dde_perts,
    generate_dte_perts,
    run_rf,
)
from atomate2.abinit.sets.core import ShgStaticSetGenerator, StaticSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

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
    mrgddb_maker : .Maker
        The maker to merge the DDE and DTE DDB.
    use_dde_sym : bool
        True if only the irreducible DDE perturbations should be considered,
            False otherwise.
    dte_skip_permutations: Since the current version of abinit always performs
        all the permutations of the perturbations, even if only one is asked,
        if True avoids the creation of inputs that will produce duplicated outputs.
    """

    name: str = "DFPT"
    static_maker: BaseAbinitMaker = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=StaticSetGenerator(factory=scf_for_phonons)
        )
    )
    ddk_maker: BaseAbinitMaker | None = field(default_factory=DdkMaker)  # |
    dde_maker: BaseAbinitMaker | None = field(
        default_factory=DdeMaker
    )  # | VT: replace by bool?
    dte_maker: BaseAbinitMaker | None = field(default_factory=DteMaker)  # |
    mrgddb_maker: Maker | None = field(default_factory=MrgddbMaker)  # |
    anaddb_maker: Maker | None = field(default_factory=AnaddbMaker)  # |
    use_dde_sym: bool = True
    dte_skip_permutations: bool | None = False

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

        Returns
        -------
        Flow
            A DFPT flow
        """
        static_job = self.static_maker.make(structure, restart_from=restart_from)

        jobs = [static_job]

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
                # next line throws :
                # File "/gpfs/home/acad/ucl-modl/vtrinque/Software/jobflow/src
                # /jobflow/utils/find.py", line 84, in _lookup
                #     if key in obj:
                # TypeError: unhashable type: 'dict'
                # with key being {'append_str': '1/3', 'prepend': False}
                # ddk_job.append_name(f"{ipert+1}/{len(perturbations)}")
                ddk_job.name = ddk_job.name + f"{ipert+1}/{len(perturbations)}"

                ddk_jobs.append(ddk_job)
                outputs["dirs"].append(ddk_job.output.dir_name)

            ddk_calcs = Flow(ddk_jobs, outputs)
            jobs.append(ddk_calcs)

        if self.dde_maker:
            # generate the perturbations for the DDE calculations
            dde_perts = generate_dde_perts(
                gsinput=static_job.output.input.abinit_input,
                use_symmetries=self.use_dde_sym,
            )
            jobs.append(dde_perts)

            # perform the DDE calculations
            dde_calcs = run_rf(
                perturbations=dde_perts.output["perts"],
                rf_maker=self.dde_maker,
                prev_outputs=[static_job.output.dir_name, ddk_calcs.output["dirs"]],
            )
            jobs.append(dde_calcs)

        if self.dte_maker:
            phonon_pert = False

            # To uncomment once there is a PhononMaker or something similar
            # if self.ph_maker:
            #     phonon_pert = True

            # generate the perturbations for the DTE calculations
            dte_perts = generate_dte_perts(
                gsinput=static_job.output.input.abinit_input,
                skip_permutations=self.dte_skip_permutations,
                phonon_pert=phonon_pert,
            )
            jobs.append(dte_perts)

            # perform the DTE calculations
            dte_calcs = run_rf(
                perturbations=dte_perts.output["perts"],
                rf_maker=self.dte_maker,
                prev_outputs=[
                    static_job.output.dir_name,
                    dde_calcs.output["dirs"],
                ],
            )
            jobs.append(dte_calcs)

        if self.mrgddb_maker:
            # merge the DDE and DTE DDB.

            prev_outputs = [dde_calcs.output["dirs"]]
            if self.dte_maker:
                prev_outputs.append(dte_calcs.output["dirs"])

            mrgddb_job = self.mrgddb_maker.make(
                prev_outputs=prev_outputs,
            )

            jobs.append(mrgddb_job)

        if self.anaddb_maker:
            # analyze a merged DDB.

            anaddb_job = self.anaddb_maker.make(
                structure=mrgddb_job.output.structure,
                prev_outputs=mrgddb_job.output.dir_name,
            )

            jobs.append(anaddb_job)

        # TODO: implement the possibility of other DFPT WFs (phonons,...)
        # if self.wfq_maker:
        #     ...

        # return Flow(jobs, output=jobs[-1].output, name=self.name)  # TODO: fix outputs
        return Flow(
            jobs, output=[j.output for j in jobs], name=self.name
        )  # TODO: fix outputs


@dataclass
class ShgFlowMaker(DfptFlowMaker):
    """
    Maker to compute the static DFPT second-harmonic generation tensor.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    """

    name: str = "DFPT Chi2 SHG"
    anaddb_maker: Maker | None = field(default_factory=AnaddbDfptDteMaker)
    use_dde_sym: bool = False
    static_maker: BaseAbinitMaker = field(
        default_factory=lambda: StaticMaker(input_set_generator=ShgStaticSetGenerator())
    )
