"""Core jobs for running ABINIT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from abipy.flowtk.events import (
    AbinitCriticalWarning,
    NscfConvergenceWarning,
    RelaxConvergenceWarning,
    ScfConvergenceWarning,
)
from jobflow import Job, job

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.sets.core import (
    LineNonSCFSetGenerator,
    NonSCFSetGenerator,
    NonScfWfqInputGenerator,
    RelaxSetGenerator,
    StaticSetGenerator,
    UniformNonSCFSetGenerator,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core.structure import Structure

    from atomate2.abinit.sets.base import AbinitInputGenerator
    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)


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
    input_set_generator: AbinitInputGenerator = field(
        default_factory=StaticSetGenerator
    )

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        ScfConvergenceWarning,
    )


@dataclass
class LineNonSCFMaker(BaseAbinitMaker):
    """Maker to create a jobs with a non-scf ABINIT calculation along a line.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "nscf_line"
    name: str = "Line non-Scf calculation"
    input_set_generator: AbinitInputGenerator = field(
        default_factory=LineNonSCFSetGenerator
    )

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
    )


@dataclass
class UniformNonSCFMaker(BaseAbinitMaker):
    """Maker to create a jobs with a non-scf ABINIT calculation along a line.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "nscf_uniform"
    name: str = "Uniform non-Scf calculation"
    input_set_generator: AbinitInputGenerator = field(
        default_factory=UniformNonSCFSetGenerator
    )

    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
    )


@dataclass
class NonSCFMaker(BaseAbinitMaker):
    """Maker to create non SCF calculations."""

    calc_type: str = "nscf"
    name: str = "non-Scf calculation"

    input_set_generator: AbinitInputGenerator = field(
        default_factory=NonSCFSetGenerator
    )

    # Non dataclass variables:
    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
    )

    @job
    def make(
        self,
        structure: Structure | None = None,
        prev_outputs: str | list[str] | None = None,
        restart_from: str | list[str] | None = None,
        history: JobHistory | None = None,
        mode: str = "uniform",
    ) -> Job:
        """Run a non-scf ABINIT job.

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

    input_set_generator: AbinitInputGenerator = field(
        default_factory=NonScfWfqInputGenerator
    )

    # Non dataclass variables:
    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        NscfConvergenceWarning,
    )


@dataclass
class RelaxMaker(BaseAbinitMaker):
    """Maker to create relaxation calculations."""

    calc_type: str = "relax"
    input_set_generator: AbinitInputGenerator = field(default_factory=RelaxSetGenerator)
    name: str = "Relaxation calculation"

    # non-dataclass variables
    CRITICAL_EVENTS: ClassVar[Sequence[AbinitCriticalWarning]] = (
        RelaxConvergenceWarning,
    )

    @classmethod
    def ionic_relaxation(cls, *args, **kwargs) -> Job:
        """Create an ionic relaxation maker."""
        # TODO: add the possibility to tune the RelaxInputGenerator options
        #  in this class method.
        return cls(
            input_set_generator=RelaxSetGenerator(*args, relax_cell=False, **kwargs),
            name=cls.name + " (ions only)",
        )

    @classmethod
    def full_relaxation(cls, *args, **kwargs) -> Job:
        """Create a full relaxation maker."""
        # TODO: add the possibility to tune the RelaxInputGenerator options
        #  in this class method.
        return cls(
            input_set_generator=RelaxSetGenerator(*args, relax_cell=True, **kwargs)
        )
