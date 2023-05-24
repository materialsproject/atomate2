"""Response function jobs for running ABINIT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ClassVar, Sequence

from jobflow import job
from pymatgen.core.structure import Structure

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.sets.base import AbinitInputGenerator
from atomate2.abinit.sets.response import (
    DdeSetGenerator,
    DdkSetGenerator,
    DteSetGenerator,
)
from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = ["DdkMaker", "DdeMaker", "DteMaker"]


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
        self.input_set_generator.factory_kwargs = {"perturbation": perturbation}

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
        self.input_set_generator.factory_kwargs = {"perturbation": perturbation}

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
        self.input_set_generator.factory_kwargs = {"perturbation": perturbation}

        return super().make.original(
            self,
            structure=structure,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
            history=history,
        )
