"""Core jobs for running JDFTx calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.jdftx.jobs.base import BaseJdftxMaker
from atomate2.jdftx.sets.core import (
    IonicMinSetGenerator,
)

if TYPE_CHECKING:
    from atomate2.jdftx.sets.base import JdftxInputGenerator

logger = logging.getLogger(__name__)

@dataclass
class SurfaceMinMaker(BaseJdftxMaker):
    """Maker to create surface relaxation job."""

    name: str = "surface_ionic_min"
    input_set_generator: JdftxInputGenerator = field(
        default_factory=IonicMinSetGenerator(
            user_settings={
                "coulomb-interaction": {
                    "truncationType": "Slab",
                    "dir": "001"
                }
            },
            auto_coulomb_truncation = True,
        )
    )

class MolMinMaker(BaseJdftxMaker):
    """Maker to create molecule relaxation job."""