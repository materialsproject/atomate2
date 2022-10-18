"""Core jobs for running ABINIT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ClassVar, Sequence

from jobflow import job
from pymatgen.core.structure import Structure

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.sets.core import (
    NonSCFSetGenerator,
    NonScfWfqInputGenerator,
    RelaxSetGenerator,
    StaticSetGenerator,
)
from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = ["StaticMaker", "NonSCFMaker", "RelaxMaker"]


@dataclass
class StaticMaker(BaseAbinitMaker):
    """Maker to create ABINIT scf jobs.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "scf"
    name: str = "Scf calculation"
    input_set_generator: StaticSetGenerator = field(default_factory=StaticSetGenerator)

    CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("ScfConvergenceWarning",)


@dataclass
class NonSCFMaker(BaseAbinitMaker):
    """Maker to create non SCF calculations."""

    calc_type: str = "nscf"
    name: str = "non-Scf calculation"

    input_set_generator: NonSCFSetGenerator = NonSCFSetGenerator()

    # Non dataclass variables:
    CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("NscfConvergenceWarning",)

    @job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | list[str] | None = None,
        restart_from: str | list[str] | None = None,
        history: JobHistory | None = None,
        mode: str = "line",
    ):
        """
        Run a non-scf ABINIT job.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        mode : str
            Type of band structure calculation. Options are:
            - "line": Full band structure along symmetry lines.
            - "uniform": Uniform mesh band structure.
        """
        self.input_set_generator.mode = mode

        return super().make.original(
            self,
            structure=structure,
            prev_outputs=prev_outputs,
            restart_from=restart_from,
            history=history,
        )


@dataclass
class NonSCFWfqMaker(NonSCFMaker):
    """Maker to create non SCF calculations for the WFQ."""

    calc_type: str = "nscf_wfq"
    name: str = "non-Scf calculation"

    input_set_generator: NonScfWfqInputGenerator = NonScfWfqInputGenerator()

    # Non dataclass variables:
    CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("NscfConvergenceWarning",)


@dataclass
class RelaxMaker(BaseAbinitMaker):
    """Maker to create relaxation calculations."""

    calc_type: str = "relax"
    input_set_generator: RelaxSetGenerator = RelaxSetGenerator()
    name: str = "Relaxation calculation"

    # non-dataclass variables
    CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("RelaxConvergenceWarning",)
    # structure_fixed: ClassVar[bool] = False

    @classmethod
    def ionic_relaxation(cls, *args, **kwargs):
        """Create an ionic relaxation maker."""
        # TODO: add the possibility to tune the RelaxInputGenerator options
        #  in this class method.
        return cls(
            input_set_generator=RelaxSetGenerator(relax_cell=False, *args, **kwargs),
            name=cls.name + " (ions only)",
        )

    @classmethod
    def full_relaxation(cls, *args, **kwargs):
        """Create a full relaxation maker."""
        # TODO: add the possibility to tune the RelaxInputGenerator options
        #  in this class method.
        return cls(
            input_set_generator=RelaxSetGenerator(relax_cell=True, *args, **kwargs)
        )
