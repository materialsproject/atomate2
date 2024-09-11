"""Core jobs for running JDFTx calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from atomate2.jdftx.jobs.base import BaseJdftxMaker
from atomate2.jdftx.sets.base import JdftxInputGenerator
from atomate2.jdftx.sets.core import BEASTSetGenerator

logger = logging.getLogger(__name__)


@dataclass
class BEASTRelaxMaker(BaseJdftxMaker):
    name: str = "relax"
    input_set_generator: JdftxInputGenerator = field(default_factory=BEASTSetGenerator)
