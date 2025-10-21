"""Jobs for running ABINIT response to perturbations."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from abipy.flowtk.events import (
    AbinitCriticalWarning,
    NscfConvergenceWarning,
    ScfConvergenceWarning,
)
from jobflow import Flow, Job, Response, job

from atomate2.abinit.jobs.base import BaseAbinitMaker, abinit_job
from atomate2.abinit.jobs.core import NonSCFMaker
from atomate2.abinit.powerups import update_user_abinit_settings
from atomate2.abinit.sets.base import get_ksampling
from atomate2.abinit.sets.response import (
    DdeSetGenerator,
    DdkSetGenerator,
    DteSetGenerator,
    NscfWfqSetGenerator,
    PhononSetGenerator,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from abipy.abio.inputs import AbinitInput
    from pymatgen.core.structure import Structure
    from pymatgen.io.abinit.abiobjects import KSampling

    from atomate2.abinit.sets.base import AbinitInputGenerator
    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = [
    "DdeMaker",
    "DdkMaker",
    "DteMaker",
    "PhononResponseMaker",
    "ResponseMaker",
    "WfqMaker",
    "generate_perts",
    "get_jobs",
]


@dataclass
class ResponseMaker(BaseAbinitMaker):
    """Maker for a Response Function ABINIT calculation job.

    Parameters
    ----------
    calc_type : str
        The type of RF.
    name : str
        The job name.
    """

    calc_type: str = "RF"
    name: str = "RF calculation"
    task_document_kwargs: dict = field(default_factory=dict)
    input_set_generator: AbinitInputGenerator
    stop_jobflow_on_failure: bool = True

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        ScfConvergenceWarning,
    )

    @abinit_job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | list[str] | None = None,
        restart_from: str | list[str] | None = None,
        history: JobHistory | None = None,
        perturbation: dict | None = None,
    ) -> Job:
        """
        Run a RF ABINIT job. The type of RF is defined by self.calc_type.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object
        perturbation : dict
            Direction of the perturbation for the RF calculation.
            Abipy format.
        """
        if perturbation:
            self.input_set_generator.factory_kwargs.update(
                {f"{self.calc_type.lower()}_pert": perturbation}
            )

        return super().make.original(
            self,
            structure=structure,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
            history=history,
        )


@dataclass
class DdkMaker(ResponseMaker):
    """Maker to create a job with a DDK ABINIT calculation.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "DDK"
    name: str = "DDK calculation"
    input_set_generator: AbinitInputGenerator = field(default_factory=DdkSetGenerator)

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
        ScfConvergenceWarning,
    )


@dataclass
class DdeMaker(ResponseMaker):
    """Maker to create a job with a DDE ABINIT calculation.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "DDE"
    name: str = "DDE calculation"
    input_set_generator: AbinitInputGenerator = field(default_factory=DdeSetGenerator)

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        ScfConvergenceWarning,
    )


@dataclass
class DteMaker(ResponseMaker):
    """Maker to create a job with a DTE ABINIT calculation.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "DTE"
    name: str = "DTE calculation"
    input_set_generator: AbinitInputGenerator = field(default_factory=DteSetGenerator)

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        ScfConvergenceWarning,
    )


@dataclass
class WfqMaker(NonSCFMaker):
    """Maker for non SCF wfq calculation."""

    calc_type: str = "wfq"
    name: str = "WFQ nscf Calculation"
    input_set_generator: AbinitInputGenerator = field(
        default_factory=NscfWfqSetGenerator
    )

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
    )

    @abinit_job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | list[str] | None = None,
        restart_from: str | list[str] | None = None,
        history: JobHistory | None = None,
        mode: str = "uniform",
        qpt: list | tuple | None = None,
    ) -> Job:
        """Run a WFQ Abinit job.

        Parameters
        ----------
        qpt: list, tuple
            q point to shift the k grid
        """
        self.input_set_generator.user_abinit_settings.update({"qpt": qpt})
        return super().make.original(
            self,
            structure=structure,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
            history=history,
            mode=mode,
        )


@dataclass
class PhononResponseMaker(ResponseMaker):
    """Maker to create a job with a Phonon ABINIT calculation.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "Phonon"
    name: str = "Phonon calculation"
    input_set_generator: AbinitInputGenerator = field(
        default_factory=PhononSetGenerator
    )

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        ScfConvergenceWarning,
    )


@job
def generate_perts(
    gsinput: AbinitInput,
    scf_output: str | list[str],
    ddk_output: str | list[str] | None = None,
    skip_dte_permutations: bool | None = False,
    use_dde_symmetries: bool | None = False,
    ngqpt: list | tuple | None = None,
    qptopt: int | None = 1,
    qpt_list: list[list] | None = None,
    user_qpoints_settings: dict | KSampling | None = None,
    dde_maker: ResponseMaker | None = None,
    wfq_maker: NonSCFMaker | None = None,
    phonon_maker: ResponseMaker | None = None,
    dte_maker: ResponseMaker | None = None,
) -> Flow:
    """
    Generate the perturbations for the DTE calculations.

    Parameters
    ----------
    gsinput : AbinitInput
        an |AbinitInput| representing a ground state calculation,
        likely the SCF performed to get the WFK.
    scf_output : str or list of str
        The output directory of the SCF calculation.
    ddk_output : str or list of str
        The output directory of the DDK calculation.
    skip_dte_permutations: Since the current version of abinit always performs
        all the permutations of the perturbations, even if only one is asked,
        if True avoids the creation of inputs that will produce duplicated outputs.
    use_dde_symmetries: bool
        True if only the irreducible DDE perturbations should be considered,
        False otherwise.
    ngqpt : list or tuple
        Monkhorst-Pack divisions for the phonon q-mesh.
        Default is the same as the one used in the GS calculation.
    qptopt : int
        Option for the generation of the q-points list, default same as kptopt in gs.
    qpt_list : list of lists
        A list of q points to compute the phonons.
    user_qpoints_settings : dict or KSampling
        Allows user to define the qmesh by supplying a dict. E.g.,
        ``{"reciprocal_density": 1000}``. User can also supply a KSampling object.
    dde_maker : ResponseMaker
        The maker to use for the DDE calculations.
    wfq_maker : NonSCFMaker
        The maker to use for the WFQ calculations.
    phonon_maker : ResponseMaker
        The maker to use for the phonon calculations.
    dte_maker : ResponseMaker
        The maker to use for the DTE calculations.

    Outputs
    -------
    rf_flow: Flow
        Flow with the response functions
    """
    if all(not m for m in [dde_maker, phonon_maker, dte_maker]):
        raise ValueError("At least one of the response makers should be defined")

    cwd = Path.cwd()
    outputs: dict = {"perts": {}, "dirs": {}}

    dde_jobs: list = []
    wfq_jobs: list = []
    ph_jobs: list = []
    dte_jobs: list = []

    if not isinstance(scf_output, list | tuple):
        scf_output = [scf_output]
    if not isinstance(ddk_output, list | tuple):
        ddk_output = [ddk_output]

    gsinput = gsinput.deepcopy()
    gsinput.pop_vars(["autoparal"])
    gsinput.pop_par_vars(all=True)

    # DDE
    if dde_maker:
        if not ddk_output:
            raise ValueError("DDK output is required to run DDE")
        if use_dde_symmetries:
            gsinput = gsinput.deepcopy()
            gsinput.pop_vars(["autoparal"])
            gsinput.pop_par_vars(all=True)
            dde_perts = gsinput.abiget_irred_ddeperts(
                workdir=cwd / "dde"
            )  # TODO: quid manager?
        else:
            dde_perts = [{"idir": 1}, {"idir": 2}, {"idir": 3}]

        dde_prev = scf_output + ddk_output
        dde_jobs = get_jobs(dde_perts, rf_maker=dde_maker, prev_outputs=dde_prev)

        outputs["perts"]["dde"] = [j.output for j in dde_jobs]
        outputs["dirs"]["dde"] = [j.output.dir_name for j in dde_jobs]

    # Phonons
    if phonon_maker:
        if (
            np.sum([x is not None for x in [ngqpt, qpt_list, user_qpoints_settings]])
            > 1
        ):
            raise ValueError(
                "qpt_list, ngqpt and user_qpoints_settings can't be used together"
            )

        if qpt_list is None:
            if ngqpt is not None:
                ngqpt = np.array(ngqpt)
            elif user_qpoints_settings is not None:
                ksampling = get_ksampling(
                    structure=gsinput.structure,
                    user_kpoints_settings=user_qpoints_settings,
                    force_gamma=True,
                )
                if "ngkpt" not in ksampling.abivars:
                    raise RuntimeError(
                        f"Could not determine ngqpt from ksampling {ksampling}"
                    )
                ngqpt = ksampling.abivars["ngkpt"]
            else:
                ngqpt = np.array(gsinput["ngkpt"])

            qpt_list = gsinput.abiget_ibz(
                ngkpt=ngqpt, shiftk=(0, 0, 0), kptopt=qptopt
            ).points

        # check that qpt are consistent with kpt grid
        outputs["dirs"]["wfq"] = [[] for _ in range(len(qpt_list))]
        if ngqpt is None or any(gsinput["ngkpt"] % ngqpt != 0):
            # find which q points are needed and build nscf inputs to calculate the WFQ
            kpts = gsinput.abiget_ibz(shiftk=(0, 0, 0), kptopt=3).points.tolist()
            nscf_qpt = []
            for qpt_i, q in enumerate(qpt_list):
                if list(q) not in kpts:
                    if wfq_maker is None:
                        raise RuntimeError(
                            "q-mesh not commensurate with k-mesh!"
                            "Either use commensurate grid or define a WFQ maker. "
                            "If this does not mean much to you, you can remove "
                            "your definition of ngqpt and/or qpt_list and "
                            "you will obtain a bandstructure on a q-mesh that is the "
                            "same as the k-mesh used for your groundstate calculation."
                        )
                    nscf_qpt.append(q)
                    wfq_job = wfq_maker.make(
                        prev_outputs=scf_output,
                        qpt=q,
                    )
                    wfq_jobs.append(wfq_job)
                    outputs["dirs"]["wfq"][qpt_i] = [
                        j.output.dir_name for j in wfq_jobs
                    ]

        ph_perts = list()
        ph_jobs = list()
        for qpt_i, q in enumerate(qpt_list):
            perts = gsinput.abiget_irred_phperts(qpt=q, workdir=cwd / f"q_{qpt_i}")
            ph_perts.extend(perts)

        ph_jobs = get_jobs(
            ph_perts,
            rf_maker=phonon_maker,
            prev_outputs=scf_output + outputs["dirs"]["wfq"][qpt_i],
        )

        outputs["perts"]["phonon"] = [j.output for j in ph_jobs]
        outputs["dirs"]["phonon"] = [j.output.dir_name for j in ph_jobs]

    # DTE
    if dte_maker:
        dte_perts = gsinput.abiget_irred_dteperts(
            phonon_pert=phonon_maker is not None, workdir=cwd / "dte"
        )  # TODO: quid manager?

        if skip_dte_permutations:
            perts_to_skip: list = []
            reduced_perts = []
            for pert in dte_perts:
                p = (
                    (pert.i1pert, pert.i1dir),
                    (pert.i2pert, pert.i2dir),
                    (pert.i3pert, pert.i3dir),
                )
                if p not in perts_to_skip:
                    reduced_perts.append(pert)
                    perts_to_skip.extend(itertools.permutations(p))

            dte_perts = reduced_perts

        dde_jobs = [
            update_user_abinit_settings(ddej, {"prtwf": 1}) for ddej in dde_jobs
        ]
        ph_jobs = [update_user_abinit_settings(pj, {"prtwf": 1}) for pj in ph_jobs]

        dte_prev = (
            scf_output
            + [j.output.dir_name for j in dde_jobs]
            + [j.output.dir_name for j in ph_jobs]
        )
        dte_jobs = get_jobs(dte_perts, rf_maker=dte_maker, prev_outputs=dte_prev)
        outputs["perts"]["dte"] = [j.output for j in dte_jobs]
        outputs["dirs"]["dte"] = [j.output.dir_name for j in dte_jobs]

    jobs = dde_jobs + wfq_jobs + ph_jobs + dte_jobs

    rf_flow = Flow(jobs, outputs)
    return Response(
        replace=rf_flow, output={"dir_name": cwd}
    )  # TODO what is the output here?


def get_jobs(
    perturbations: list[dict],
    rf_maker: ResponseMaker,
    prev_outputs: str | list[str] | None = None,
) -> list[Job]:
    """
    Set up the response jobs for each perturbations.

    Parameters
    ----------
    perturbations : a list of dict with the direction of the perturbation
        under the Abipy format.
    rf_maker : Maker to create a job with a Response Function ABINIT calculation.
    prev_outputs : a list of previous output directories
    is_phonon: whether this is a phonon calculation
    """
    rf_jobs = []

    for ipert, pert in enumerate(perturbations):
        rf_job = rf_maker.make(
            perturbation=pert,
            prev_outputs=prev_outputs,
        )

        rf_job.append_name(f"{ipert + 1}/{len(perturbations)}")

        rf_jobs.append(rf_job)

    return rf_jobs
