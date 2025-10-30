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
from jobflow.core.reference import OutputReference

from atomate2.abinit.jobs.base import BaseAbinitMaker, abinit_job
from atomate2.abinit.powerups import update_user_abinit_settings
from atomate2.abinit.sets.base import get_ksampling
from atomate2.abinit.sets.response import (
    DdeSetGenerator,
    DdkSetGenerator,
    DteSetGenerator,
    PhononSetGenerator,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from abipy.abio.inputs import AbinitInput
    from pymatgen.core.structure import Structure
    from pymatgen.io.abinit.abiobjects import KSampling

    from atomate2.abinit.jobs.core import NonSCFMaker
    from atomate2.abinit.sets.base import AbinitInputGenerator
    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = [
    "DdeMaker",
    "DdkMaker",
    "DteMaker",
    "PhononResponseMaker",
    "ResponseMaker",
    "generate_perts",
    "get_jobs",
]


@dataclass
class ResponseMaker(BaseAbinitMaker):
    """
    Base maker for DFPT response function calculations.

    This is the base class for all DFPT response function makers including
    DDK, DDE, DTE, and phonon perturbations. It handles setting up the
    perturbation direction for the calculation.

    Parameters
    ----------
    calc_type : str
        The type of response function calculation. Default is "RF".
    name : str
        The job name. Default is "RF calculation".
    task_document_kwargs : dict
        Additional keyword arguments passed to TaskDoc.from_directory().
    input_set_generator : AbinitInputGenerator
        Generator for ABINIT input files. Must be provided by subclasses.
    stop_jobflow_on_failure : bool
        If True, stop the entire jobflow when this job fails. Default is True.
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
        Create a response function ABINIT job.

        The type of response function is determined by self.calc_type (DDK,
        DDE, DTE, or Phonon).

        Parameters
        ----------
        structure : Structure or None
            A pymatgen Structure object. At least one of structure,
            prev_outputs, or restart_from must be provided.
        prev_outputs : str or list[str] or None
            Path(s) to previous calculation directories to use as inputs.
        restart_from : str or list[str] or None
            Path(s) to previous calculation directories to restart from.
        history : JobHistory or None
            Job history tracking previous runs and restarts.
        perturbation : dict or None
            Direction of the perturbation for the response function
            calculation, in AbiPy format (e.g., {"idir": 1}).

        Returns
        -------
        Job
            A jobflow Job for the response function calculation.
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
    """
    Maker for DDK (derivative of wavefunctions with respect to k) calculations.

    DDK calculations compute the derivative of wavefunctions with respect to
    the k-point, which is required for electric field perturbations (DDE).

    Parameters
    ----------
    calc_type : str
        The calculation type identifier. Default is "DDK".
    name : str
        The job name. Default is "DDK calculation".
    input_set_generator : AbinitInputGenerator
        Generator for ABINIT input files. Defaults to DdkSetGenerator.
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
    """
    Maker for DDE (derivative with respect to electric field) calculations.

    DDE calculations compute the response to an electric field perturbation,
    yielding Born effective charges and the electronic dielectric tensor.
    Requires DDK calculations as input.

    Parameters
    ----------
    calc_type : str
        The calculation type identifier. Default is "DDE".
    name : str
        The job name. Default is "DDE calculation".
    input_set_generator : AbinitInputGenerator
        Generator for ABINIT input files. Defaults to DdeSetGenerator.
    """

    calc_type: str = "DDE"
    name: str = "DDE calculation"
    input_set_generator: AbinitInputGenerator = field(default_factory=DdeSetGenerator)

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        ScfConvergenceWarning,
    )


@dataclass
class DteMaker(ResponseMaker):
    """
    Maker for DTE (mixed derivative) calculations.

    DTE calculations compute the response to combined perturbations, yielding properties
    like static SHG and elastic tensors.
    Requires both DDE and phonon calculations as input.

    Parameters
    ----------
    calc_type : str
        The calculation type identifier. Default is "DTE".
    name : str
        The job name. Default is "DTE calculation".
    input_set_generator : AbinitInputGenerator
        Generator for ABINIT input files. Defaults to DteSetGenerator.
    """

    calc_type: str = "DTE"
    name: str = "DTE calculation"
    input_set_generator: AbinitInputGenerator = field(default_factory=DteSetGenerator)

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        ScfConvergenceWarning,
    )


@dataclass
class PhononResponseMaker(ResponseMaker):
    """
    Maker for phonon response calculations.

    Phonon calculations compute the response to atomic displacement
    perturbations at specific q-points, yielding the dynamical matrix
    and phonon frequencies.

    Parameters
    ----------
    calc_type : str
        The calculation type identifier. Default is "Phonon".
    name : str
        The job name. Default is "Phonon calculation".
    input_set_generator : AbinitInputGenerator
        Generator for ABINIT input files. Defaults to PhononSetGenerator.
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
    Generate DFPT perturbation calculations for response properties.

    This function creates a flow of DFPT calculations including DDE (electric
    field), phonon (atomic displacement), and DTE (mixed) perturbations. It
    automatically determines which perturbations are needed, handles symmetries,
    and creates WFQ calculations when q-points are incommensurate with the
    k-point grid.

    Parameters
    ----------
    gsinput : AbinitInput
        An AbinitInput object representing the ground state calculation,
        typically the SCF calculation used to generate the WFK file.
    scf_output : str or list[str]
        Path(s) to the output directory of the SCF calculation.
    ddk_output : str or list[str] or None
        Path(s) to the output directory of the DDK calculation. Required
        if dde_maker is provided.
    skip_dte_permutations : bool or None
        If True, skip creation of redundant DTE inputs. Since ABINIT always
        computes all permutations of perturbations even when only one is
        requested, this avoids duplicated outputs. Default is False.
    use_dde_symmetries : bool or None
        If True, use symmetries to reduce DDE perturbations to only
        irreducible ones. Default is False.
    ngqpt : list or tuple or None
        Monkhorst-Pack divisions for the phonon q-point mesh (e.g., [4, 4, 4]).
        If None, uses the same grid as the ground state k-points.
    qptopt : int or None
        Option for q-point generation. Default is 1 (same as kptopt in ground state).
    qpt_list : list[list] or None
        Explicit list of q-points for phonon calculations. Cannot be used
        with ngqpt or user_qpoints_settings.
    user_qpoints_settings : dict or KSampling or None
        Custom q-point settings (e.g., {"reciprocal_density": 1000}) or a
        KSampling object. Cannot be used with ngqpt or qpt_list.
    dde_maker : ResponseMaker or None
        Maker for DDE (electric field) calculations. If None, DDE
        calculations are skipped.
    wfq_maker : NonSCFMaker or None
        Maker for WFQ calculations. Required if the q-point mesh is not
        commensurate with the k-point mesh.
    phonon_maker : ResponseMaker or None
        Maker for phonon (atomic displacement) calculations. If None,
        phonon calculations are skipped.
    dte_maker : ResponseMaker or None
        Maker for DTE (mixed derivative) calculations. If None, DTE
        calculations are skipped.

    Returns
    -------
    Flow
        A jobflow Flow containing all perturbation calculations, with outputs
        organized by perturbation type (dde, phonon, dte, wfq).

    Raises
    ------
    ValueError
        If no response makers are provided, or if DDK output is missing when
        dde_maker is provided, or if multiple q-point specification methods
        are used simultaneously.
    RuntimeError
        If q-mesh is not commensurate with k-mesh and no wfq_maker is provided.
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
            dde_perts = gsinput.abiget_irred_ddeperts(workdir=cwd / "dde")
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

        # Check if q-mesh is commensurate with k-mesh, create WFQ jobs if needed
        outputs["dirs"]["wfq"] = [[] for _ in range(len(qpt_list))]
        if ngqpt is None or any(gsinput["ngkpt"] % ngqpt != 0):
            # Find incommensurate q-points and create WFQ calculations for them
            kpts = gsinput.abiget_ibz(shiftk=(0, 0, 0), kptopt=3).points.tolist()
            nscf_qpt = []
            wfq_j = 0
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
                    wfq_j += 1
                    nscf_qpt.append(q)
                    wfq_job = wfq_maker.make(
                        prev_outputs=scf_output,
                        qpt=q,
                    )
                    wfq_job.append_name(f" - {wfq_j}")
                    wfq_jobs.append(wfq_job)
                    outputs["dirs"]["wfq"][qpt_i] = [wfq_job.output.dir_name]

        ph_perts = list()
        ph_jobs = list()
        prev_outputs: list[str] = list()
        for qpt_i, q in enumerate(qpt_list):
            perts = gsinput.abiget_irred_phperts(qpt=q, workdir=cwd / f"q_{qpt_i}")
            ph_perts.extend(perts)
            prev_outputs.extend(
                scf_output + outputs["dirs"]["wfq"][qpt_i] for _ in range(len(perts))
            )

        ph_jobs = get_jobs(
            ph_perts,
            rf_maker=phonon_maker,
            prev_outputs=prev_outputs,
        )

        outputs["perts"]["phonon"] = [j.output for j in ph_jobs]
        outputs["dirs"]["phonon"] = [j.output.dir_name for j in ph_jobs]

    # DTE
    if dte_maker:
        dte_perts = gsinput.abiget_irred_dteperts(
            phonon_pert=phonon_maker is not None, workdir=cwd / "dte"
        )

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
    return Response(replace=rf_flow, output={"dir_name": cwd})


def get_jobs(
    perturbations: list[dict],
    rf_maker: ResponseMaker,
    prev_outputs: str | OutputReference | list[str] | list[list[str]] | None = None,
) -> list[Job]:
    """
    Create response function jobs for multiple perturbations.

    This function creates a list of jobs for response function calculations,
    one for each perturbation direction. It handles different formats of
    prev_outputs (single path, list of paths, or list of lists).

    Parameters
    ----------
    perturbations : list[dict]
        List of perturbation dictionaries in AbiPy format. Each dict
        specifies the direction and type of perturbation (e.g., {"idir": 1}).
    rf_maker : ResponseMaker
        Maker to create response function ABINIT jobs. Can be any subclass
        of ResponseMaker (DdkMaker, DdeMaker, PhononResponseMaker, etc.).
    prev_outputs : str or OutputReference or list[str] or list[list[str]] or None
        Previous output directories to use as inputs. Can be:
        - A single path (str or OutputReference): used for all perturbations
        - A list of paths: used for all perturbations
        - A list of lists: each inner list used for the corresponding perturbation
          (must have same length as perturbations)

    Returns
    -------
    list[Job]
        List of response function jobs, one per perturbation, with names
        appended to indicate progress (e.g., " - 1/3", " - 2/3").

    Raises
    ------
    ValueError
        If prev_outputs is a list of lists but its length doesn't match the
        number of perturbations, or if prev_outputs format is invalid.
    """
    rf_jobs = []

    if isinstance(prev_outputs, str | OutputReference):
        prev_outputs = [[prev_outputs] for _ in range(len(perturbations))]
    elif isinstance(prev_outputs, list) and all(
        isinstance(i, str | OutputReference) for i in prev_outputs
    ):
        prev_outputs = [prev_outputs for _ in range(len(perturbations))]
    elif isinstance(prev_outputs, list) and all(
        isinstance(i, list) and all(isinstance(j, str | OutputReference) for j in i)
        for i in prev_outputs
    ):
        if len(prev_outputs) != len(perturbations):
            raise ValueError(
                "If a list of lists is passed as prev_outputs "
                "its length must match the number of perturbations"
            )
    else:
        raise ValueError(
            "Please provide the prev_outputs as str or list[str] or list[list[str]]"
        )

    for ipert, (pert, prev_output) in enumerate(
        zip(perturbations, prev_outputs, strict=False)
    ):
        rf_job = rf_maker.make(
            perturbation=pert,
            prev_outputs=prev_output,
        )

        rf_job.append_name(f" - {ipert + 1}/{len(perturbations)}", dynamic=False)

        rf_jobs.append(rf_job)

    return rf_jobs
