"""Core jobs for running ABINIT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar, Sequence

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.sets.core import (
    NonSCFSetGenerator,
    NonScfWfqInputGenerator,
    RelaxSetGenerator,
    StaticSetGenerator,
)

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
    input_set_generator: StaticSetGenerator = StaticSetGenerator()

    CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("ScfConvergenceWarning",)


@dataclass
class NonSCFMaker(BaseAbinitMaker):
    """Maker to create non SCF calculations."""

    calc_type: str = "nscf"
    name: str = "non-Scf calculation"

    input_set_generator: NonSCFSetGenerator = NonSCFSetGenerator()

    # Non dataclass variables:
    CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("NscfConvergenceWarning",)


@dataclass
class NonSCFWfqMaker(NonSCFMaker):
    """Maker to create non SCF calculations for the WFQ."""

    calc_type: str = "nscf_wfq"
    name: str = "non-Scf calculation"

    input_set_generator: NonScfWfqInputGenerator = NonScfWfqInputGenerator()

    wfq_tolwfr: float = 1.0e-22

    # Non dataclass variables:
    CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("NscfConvergenceWarning",)
    restart_extension = "WFQ"


@dataclass
class RelaxMaker(BaseAbinitMaker):
    """Maker to create relaxation calculations."""

    calc_type: str = "relax"
    input_set_generator: RelaxSetGenerator = RelaxSetGenerator()
    name: str = "Relaxation calculation"

    # non-dataclass variables
    CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("RelaxConvergenceWarning",)
    structure_fixed: ClassVar[bool] = False

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
