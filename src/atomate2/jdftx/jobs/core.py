"""Core jobs for running JDFTx calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.jdftx.jobs.base import BaseJdftxMaker
from atomate2.jdftx.sets.core import (
    IonicMinSetGenerator,
    LatticeMinSetGenerator,
    SinglePointSetGenerator,
)

if TYPE_CHECKING:
    from atomate2.jdftx.sets.base import JdftxInputGenerator


logger = logging.getLogger(__name__)


@dataclass
class SinglePointMaker(BaseJdftxMaker):
    """Maker to create JDFTx ionic optimization job."""

    name: str = "single_point"
    input_set_generator: JdftxInputGenerator = field(
        default_factory=SinglePointSetGenerator
    )


@dataclass
class IonicMinMaker(BaseJdftxMaker):
    """Maker to create JDFTx ionic optimization job."""

    name: str = "ionic_min"
    input_set_generator: JdftxInputGenerator = field(
        default_factory=IonicMinSetGenerator
    )


@dataclass
class LatticeMinMaker(BaseJdftxMaker):
    """Maker to create JDFTx lattice optimization job."""

    name: str = "lattice_min"
    input_set_generator: JdftxInputGenerator = field(
        default_factory=LatticeMinSetGenerator
    )
