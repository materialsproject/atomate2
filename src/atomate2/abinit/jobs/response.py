"""Response function jobs for running ABINIT calculations."""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass, field
from typing import ClassVar, Sequence

from abipy.abio.inputs import AbinitInput
from abipy.flowtk.utils import Directory
from jobflow import Flow, Response, job
from pymatgen.core.structure import Structure

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.powerups import update_user_abinit_settings
from atomate2.abinit.sets.base import AbinitInputGenerator
from atomate2.abinit.sets.response import (
    DdeSetGenerator,
    DdkSetGenerator,
    DteSetGenerator,
)
from atomate2.abinit.utils.common import OUTDIR_NAME
from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = [
    "DdkMaker",
    "DdeMaker",
    "DteMaker",
    "generate_ddk_perts",
    "generate_dde_perts",
    "generate_dte_perts",
    "run_ddk_rf",
    "run_dde_rf",
    "run_dte_rf",
]


@dataclass
class DdkMaker(BaseAbinitMaker):
    """Maker to create a job with a DDK ABINIT calculation.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "DDK"
    name: str = "DDK calculation"
    input_set_generator: AbinitInputGenerator = field(default_factory=DdkSetGenerator)

    CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("NscfConvergenceWarning",)

    @job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | list[str] | None = None,
        restart_from: str | list[str] | None = None,
        history: JobHistory | None = None,
        perturbation: dict | None = None,
    ):
        """
        Run a DDK ABINIT job.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object
        perturbation : dict
            Direction of the perturbation for the DDK calculation.
            Abipy format.
        """
        self.input_set_generator.factory_kwargs = {"ddk_pert": perturbation}

        return super().make.original(
            self,
            structure=structure,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
            history=history,
        )


@dataclass
class DdeMaker(BaseAbinitMaker):
    """Maker to create a job with a DDE ABINIT calculation.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "DDE"
    name: str = "DDE calculation"
    input_set_generator: AbinitInputGenerator = field(default_factory=DdeSetGenerator)

    CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("ScfConvergenceWarning",)

    @job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | list[str] | None = None,
        restart_from: str | list[str] | None = None,
        history: JobHistory | None = None,
        perturbation: dict | None = None,
    ):
        """
        Run a DDE ABINIT job.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object
        perturbation : dict
            Direction of the perturbation for the DDE calculation.
            Abipy format.
        """
        self.input_set_generator.factory_kwargs = {"dde_pert": perturbation}

        return super().make.original(
            self,
            structure=structure,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
            history=history,
        )


@dataclass
class DteMaker(BaseAbinitMaker):
    """Maker to create a job with a DTE ABINIT calculation.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "DTE"
    name: str = "DTE calculation"
    input_set_generator: AbinitInputGenerator = field(default_factory=DteSetGenerator)

    CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("ScfConvergenceWarning",)

    @job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | list[str] | None = None,
        restart_from: str | list[str] | None = None,
        history: JobHistory | None = None,
        perturbation: dict | None = None,
    ):
        """
        Run a DTE ABINIT job.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object
        perturbation : dict
            Direction of the perturbation for the DTE calculation.
            Abipy format.
        """
        self.input_set_generator.factory_kwargs = {"dte_pert": perturbation}

        return super().make.original(
            self,
            structure=structure,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
            history=history,
        )


@job
def generate_ddk_perts(
    gsinput: AbinitInput = field(default_factory=AbinitInput),
    # TODO: or gsinput via prev_outputs?
    use_symmetries: bool | None = False,
    qpt: list | None = None,
):
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
    gsinput: AbinitInput = field(default_factory=AbinitInput),
    # TODO: or gsinput via prev_outputs?
    use_symmetries: bool | None = False,
):
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
    gsinput: AbinitInput = field(default_factory=AbinitInput),
    # TODO: or gsinput via prev_outputs?
    skip_permutations: bool | None = False,
    phonon_pert: bool | None = False,
    ixc: int | None = None,
):
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
def run_ddk_rf(
    perturbations: list[dict],
    ddk_maker: BaseAbinitMaker
    | None = DdkMaker(),  # field(default_factory=DdkMaker), removed because "'Field' object has no attribute 'make'"
    prev_outputs: list[str] | None = None,
    structure: Structure | None = None,
):
    """
    Run the DDK calculations.

    Parameters
    ----------
    perturbations : a list of dict with the direction of the perturbation
        under the Abipy format.
    prev_outputs : a list of previous output directories
    gsinput : an |AbinitInput| representing a ground state calculation,
        likely the SCF performed to get the WFK.
    ddk_maker : Maker to create a job with a DDK ABINIT calculation.
    """
    if ddk_maker is None:
        ddk_maker = DdkMaker()

    ddk_jobs = []
    outputs: dict[str, list] = {"dirs": []}

    for ipert, pert in enumerate(perturbations):
        ddk_job = ddk_maker.make(
            perturbation=pert,
            prev_outputs=prev_outputs,
            # structure=structure, removed because of
            #   factory_prev_inputs_kwargs already used
        )
        ddk_job.append_name(f"{ipert+1}/{len(perturbations)}")
        #ddk_job = update_user_abinit_settings(ddk_job, {'tolwfr': 1e-8}) #VT TO REMOVE, ONLY FOR TESTING

        ddk_jobs.append(ddk_job)
        outputs["dirs"].append(ddk_job.output.dir_name)  # TODO: determine outputs

    ddk_flow = Flow(ddk_jobs, outputs)
    return Response(replace=ddk_flow)


@job
def run_dde_rf(
    perturbations: list[dict],
    dde_maker: BaseAbinitMaker | None = DdeMaker(), # field(default_factory=DdeMaker),
    prev_outputs: list[str] | None = None,
    structure: Structure | None = None,
):
    """
    Run the DDE calculations.

    Parameters
    ----------
    perturbations : a list of dict with the direction of the perturbation
        under the Abipy format.
    prev_outputs : a list of previous output directories
    gsinput : an |AbinitInput| representing a ground state calculation,
        likely the SCF performed to get the WFK.
    dde_maker : Maker to create a job with a DDE ABINIT calculation.
    """
    if dde_maker is None:
        dde_maker = DdeMaker()

    dde_jobs = []
    outputs: dict[str, list] = {"dirs": []}

    prev_outputs = [item for sublist in prev_outputs for item in sublist]
    # Create symlink out_DDK pointing to out_1WF... to force the use of irdddk
    # instead of ird1wf later on
    #for ddk_dir_path in prev_outputs[1:]:
    #    ddk_dir_out = Directory(os.path.join(ddk_dir_path, OUTDIR_NAME))
    #    ddk_dir_out.symlink_abiext('1WF', 'DDK')

    for ipert, pert in enumerate(perturbations):
        dde_job = dde_maker.make(
            perturbation=pert,
            prev_outputs=prev_outputs,
            #structure=structure,
        )
        dde_job.append_name(f"{ipert+1}/{len(perturbations)}")
        dde_job = update_user_abinit_settings(dde_job, {'irdddk': 1, 'ird1wf': 0})#, 'tolvrs': 1e-8}) #VT TO REMOVE TOLVRS ONLY FOR TESTING

        dde_jobs.append(dde_job)
        outputs["dirs"].append(dde_job.output.dir_name)  # TODO: determine outputs

    dde_flow = Flow(dde_jobs, outputs)
    return Response(replace=dde_flow)


@job
def run_dte_rf(
    perturbations: list[dict],
    dte_maker: BaseAbinitMaker | None = DteMaker(), # field(default_factory=DteMaker),
    prev_outputs: list[str] | None = None,
    structure: Structure | None = None,
):
    """
    Run the DTE calculations.

    Parameters
    ----------
    perturbations : a list of dict with the direction of the perturbation
        under the Abipy format.
    prev_outputs : a list of previous output directories
    gsinput : an |AbinitInput| representing a ground state calculation,
        likely the SCF performed to get the WFK.
    dte_maker : Maker to create a job with a DTE ABINIT calculation.
    """
    if dte_maker is None:
        dte_maker = DteMaker()

    dte_jobs = []
    outputs: dict[str, list] = {"dirs": []}

    prev_outputs = [item for sublist in prev_outputs for item in sublist]

    for ipert, pert in enumerate(perturbations):
        dte_job = dte_maker.make(
            perturbation=pert,
            prev_outputs=prev_outputs,
            #structure=structure,
        )
        dte_job.append_name(f"{ipert+1}/{len(perturbations)}")

        dte_jobs.append(dte_job)
        outputs["dirs"].append(dte_job.output.dir_name)  # TODO: determine outputs

    dte_flow = Flow(dte_jobs, outputs)
    return Response(replace=dte_flow)

@job
def anaddb_nlo(
    prev_outputs: list[str]
):
    """
    Analyzes the final DDB (anaddb) to retrieve the SHG tensor and the
    dielectric tensor.
    """
    prev_output = prev_outputs + ''
