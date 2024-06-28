"""Response function jobs for running ABINIT calculations."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from abipy.flowtk.events import (
    AbinitCriticalWarning,
    NscfConvergenceWarning,
    ScfConvergenceWarning,
)
from jobflow import Flow, Job, Response, job

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.powerups import update_user_abinit_settings
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
    "ResponseMaker",
    "DdkMaker",
    "DdeMaker",
    "DteMaker",
    "generate_ddk_perts",
    "generate_dde_perts",
    "generate_dte_perts",
    "run_rf",
]


@dataclass
class ResponseMaker(BaseAbinitMaker):
    """Maker to create a job with a Response Function ABINIT calculation.
        The type of RF is defined by the self.calc_type attribute.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str | None = None
    name: str = "RF calculation"
    input_set_generator: AbinitInputGenerator | None = None

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        ScfConvergenceWarning,
    )

    @job
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
def generate_ddk_perts(
    gsinput: AbinitInput,
    # TODO: or gsinput via prev_outputs?
    use_symmetries: bool | None = False,
    qpt: list | None = None,
) -> list[dict[str, int]]:
    """
    Generate the perturbations for the DDK calculations.

    Parameters
    ----------
    gsinput : an |AbinitInput| representing a ground state calculation,
        likely the SCF performed to get the WFK.
    use_symmetries : True if only the irreducible perturbations should
        be returned, False otherwise.
    qpt: qpoint of the phonon in reduced coordinates. Used to shift the k-mesh
        if qpt is not passed, gsinput must already contain "qpt"
        otherwise an exception is raised.
    """
    if use_symmetries:
        perts = gsinput.abiget_irred_phperts(qpt=qpt)  # TODO: quid manager?
    else:
        perts = [{"idir": 1}, {"idir": 2}, {"idir": 3}]

    return perts


@job
def generate_dde_perts(
    gsinput: AbinitInput,
    # TODO: or gsinput via prev_outputs?
    use_symmetries: bool | None = False,
) -> list[dict[str, int]]:
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
        perts = gsinput.abiget_irred_ddeperts()  # TODO: quid manager?
    else:
        perts = [{"idir": 1}, {"idir": 2}, {"idir": 3}]

    return perts


@job
def generate_dte_perts(
    gsinput: AbinitInput,
    # TODO: or gsinput via prev_outputs?
    skip_permutations: bool | None = False,
    phonon_pert: bool | None = False,
    ixc: int | None = None,
) -> list[dict[str, int]]:
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
    ixc: Value of ixc variable. Used to overwrite the default value read
        from pseudos.
    """
    # Call Abinit to get the list of irreducible perturbations
    gsinput = gsinput.deepcopy()
    gsinput.pop_vars(["autoparal", "npfft"])
    perts = gsinput.abiget_irred_dteperts(
        phonon_pert=phonon_pert, ixc=ixc
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

    return perts


@job
def run_rf(
    perturbations: list[dict],
    rf_maker: BaseAbinitMaker = field(
        default_factory=BaseAbinitMaker
    ),  # TODO: change to generic ResponseMaker
    prev_outputs: list[str] | None = None,
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
    outputs: dict[str, list] = {"dirs": []}

    if isinstance(rf_maker, DdeMaker) or isinstance(rf_maker, DteMaker):
        prev_outputs = [item for sublist in prev_outputs for item in sublist]

    for ipert, pert in enumerate(perturbations):
        rf_job = rf_maker.make(
            perturbation=pert,
            prev_outputs=prev_outputs,
        )
        rf_job.append_name(f"{ipert+1}/{len(perturbations)}")

        if isinstance(rf_maker, DdeMaker):
            rf_job = update_user_abinit_settings(rf_job, {"irdddk": 1, "ird1wf": 0})

        rf_jobs.append(rf_job)
        outputs["dirs"].append(rf_job.output.dir_name)  # TODO: determine outputs

    rf_flow = Flow(rf_jobs, outputs)
    return Response(replace=rf_flow)
