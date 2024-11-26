"""Jobs for running ABINIT response to perturbations."""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from abipy.flowtk.events import (
    AbinitCriticalWarning,
    NscfConvergenceWarning,
    ScfConvergenceWarning,
)
from jobflow import Flow, Job, Response, job

from atomate2.abinit.jobs.base import BaseAbinitMaker, abinit_job
from atomate2.abinit.sets.response import (
    DdeSetGenerator,
    DdkSetGenerator,
    DteSetGenerator,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from abipy.abio.inputs import AbinitInput
    from pymatgen.core.structure import Structure

    from atomate2.abinit.sets.base import AbinitInputGenerator
    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = [
    "DdeMaker",
    "DdkMaker",
    "DteMaker",
    "ResponseMaker",
    "generate_dde_perts",
    "generate_dte_perts",
    "run_rf",
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
    task_document_kwargs: dict = field(
        default_factory=lambda: {"files_to_store": ["DDB"]}
    )
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


@job
def generate_dde_perts(
    gsinput: AbinitInput,
    # TODO: or gsinput via prev_outputs?
    use_symmetries: bool | None = False,
) -> dict[str, dict]:
    """
    Generate the perturbations for the DDE calculations.

    Parameters
    ----------
    gsinput : an |AbinitInput| representing a ground state calculation,
        likely the SCF performed to get the WFK.
    use_symmetries : True if only the irreducible perturbations should
        be returned, False otherwise.
    """
    if use_symmetries:
        gsinput = gsinput.deepcopy()
        gsinput.pop_vars(["autoparal"])
        gsinput.pop_par_vars(all=True)
        perts = gsinput.abiget_irred_ddeperts()  # TODO: quid manager?
    else:
        perts = [{"idir": 1}, {"idir": 2}, {"idir": 3}]

    outputs = {"perts": perts}
    # to make the dir of generate... accessible
    outputs["dir_name"] = Path(os.getcwd())
    return outputs


@job
def generate_dte_perts(
    gsinput: AbinitInput,
    # TODO: or gsinput via prev_outputs?
    skip_permutations: bool | None = False,
    phonon_pert: bool | None = False,
) -> dict[str, dict]:
    """
    Generate the perturbations for the DTE calculations.

    Parameters
    ----------
    gsinput : an |AbinitInput| representing a ground state calculation,
        likely the SCF performed to get the WFK.
    skip_permutations: Since the current version of abinit always performs
        all the permutations of the perturbations, even if only one is asked,
        if True avoids the creation of inputs that will produce duplicated outputs.
    phonon_pert: is True also the phonon perturbations will be considered.
        Default False.
    """
    # Call Abinit to get the list of irreducible perturbations
    gsinput = gsinput.deepcopy()
    gsinput.pop_vars(["autoparal"])
    gsinput.pop_par_vars(all=True)
    perts = gsinput.abiget_irred_dteperts(
        phonon_pert=phonon_pert,
    )  # TODO: quid manager?

    if skip_permutations:
        perts_to_skip: list = []
        reduced_perts = []
        for pert in perts:
            p = (
                (pert.i1pert, pert.i1dir),
                (pert.i2pert, pert.i2dir),
                (pert.i3pert, pert.i3dir),
            )
            if p not in perts_to_skip:
                reduced_perts.append(pert)
                perts_to_skip.extend(itertools.permutations(p))

        perts = reduced_perts

    outputs = {"perts": perts}
    outputs["dir_name"] = Path(os.getcwd())  # to make the dir of run_rf accessible
    return outputs


@job
def run_rf(
    perturbations: list[dict],
    rf_maker: ResponseMaker,
    prev_outputs: str | list[str] | None = None,
) -> Flow:
    """
    Run the RF calculations.

    Parameters
    ----------
    perturbations : a list of dict with the direction of the perturbation
        under the Abipy format.
    rf_maker : Maker to create a job with a Response Function ABINIT calculation.
    prev_outputs : a list of previous output directories
    """
    rf_jobs = []
    outputs: dict[str, Any] = {"dirs": []}

    if isinstance(rf_maker, DdeMaker | DteMaker):
        # Flatten the list of previous outputs dir
        # prev_outputs = [item for sublist in prev_outputs for item in sublist]
        prev_outputs = list(np.hstack(prev_outputs))

    for ipert, pert in enumerate(perturbations):
        rf_job = rf_maker.make(
            perturbation=pert,
            prev_outputs=prev_outputs,
        )
        rf_job.append_name(f"{ipert+1}/{len(perturbations)}")

        rf_jobs.append(rf_job)
        outputs["dirs"].append(rf_job.output.dir_name)  # TODO: determine outputs

    outputs["dir_name"] = Path(os.getcwd())  # to make the dir of run_rf accessible
    rf_flow = Flow(rf_jobs, outputs)

    return Response(replace=rf_flow)
