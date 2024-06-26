"""Core jobs for running ABINIT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import numpy as np
from jobflow import Response, job

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.sets.gw import ScreeningSetGenerator, SigmaSetGenerator

logger = logging.getLogger(__name__)

__all__ = ["ScreeningMaker", "SigmaMaker"]


@dataclass
class ScreeningMaker(BaseAbinitMaker):
    """Maker to create ABINIT scf jobs.

    Parameters
    ----------
    name : str
        The job name.
    """

    calc_type: str = "scr"
    name: str = "Screening calculation"

    input_set_generator: ScreeningSetGenerator = field(
        default_factory=ScreeningSetGenerator
    )

    # CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("ScfConvergenceWarning",)


@dataclass
class SigmaMaker(BaseAbinitMaker):
    """Maker to create non SCF calculations."""

    calc_type: str = "sigma"
    name: str = "Sigma calculation"

    input_set_generator: SigmaSetGenerator = field(
        default_factory=SigmaSetGenerator
    )

    # CRITICAL_EVENTS: ClassVar[Sequence[str]] = ("ScfConvergenceWarning",)


