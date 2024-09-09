"""Core jobs for running JDFTx calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.alchemy.transmuters import StandardTransmuter

from atomate2.common.utils import get_transformations

from atomate2.jdftx.jobs.base import BaseJdftxMaker
from atomate2.jdftx.sets.base import JdftxInputGenerator
from atomate2.jdftx.sets.core import BEASTSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Response
    from pymatgen.core.structure import Structure

    from atomate2.vasp.sets.base import VaspInputGenerator


logger = logging.getLogger(__name__)

@dataclass
class BEASTRelaxMaker(BaseJdftxMaker):

    name: str = "relax"
    input_set_generator: JdftxInputGenerator = field(default_factory=BEASTSetGenerator)