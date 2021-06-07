"""Core jobs for running VASP calculations."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from atomate2.vasp.jobs.base import BaseVaspMaker

if typing.TYPE_CHECKING:
    pass


@dataclass
class RelaxMaker(BaseVaspMaker):
    """Maker to create VASP relaxation jobs."""

    name: str = "structure relaxation"
    input_set: str = "MPRelaxSet"


@dataclass
class StaticMaker(BaseVaspMaker):
    """Maker to create VASP static jobs."""

    name: str = "structure relaxation"
    input_set: str = "MPStaticSet"
