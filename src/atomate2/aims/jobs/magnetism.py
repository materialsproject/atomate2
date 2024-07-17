"""Define Makers for Magnetic ordering flow in FHI-aims."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pymatgen.io.aims.sets.magnetism import (
    MagneticRelaxSetGenerator,
    MagneticStaticSetGenerator,
)

if TYPE_CHECKING:
    from pymatgen.io.aims.sets.base import AimsInputGenerator


from atomate2.aims.jobs.core import RelaxMaker, StaticMaker


@dataclass
class MagneticStaticMaker(StaticMaker):
    """Maker to create FHI-aims SCF jobs.

    Parameters
    ----------
    calc_type: str
        The type key for the calculation
    name: str
        The job name
    input_set_generator: .AimsInputGenerator
        The InputGenerator for the calculation
    """

    calc_type: str = "magnetic_scf"
    name: str = "Magnetic SCF Calculation"
    input_set_generator: AimsInputGenerator = field(
        default_factory=MagneticStaticSetGenerator
    )


@dataclass
class MagneticRelaxMaker(RelaxMaker):
    """Maker to create relaxation calculations.

    Parameters
    ----------
    calc_type: str
        The type key for the calculation
    name: str
        The job name
    input_set_generator: .AimsInputGenerator
        The InputGenerator for the calculation
    """

    calc_type: str = "relax"
    name: str = "Magnetic Relaxation calculation"
    input_set_generator: AimsInputGenerator = field(
        default_factory=MagneticRelaxSetGenerator
    )

    @classmethod
    def fixed_cell_relaxation(cls, *args, **kwargs) -> RelaxMaker:
        """Create a fixed cell relaxation maker."""
        return cls(
            input_set_generator=MagneticRelaxSetGenerator(
                *args, relax_cell=False, **kwargs
            ),
            name=cls.name + " (fixed cell)",
        )

    @classmethod
    def full_relaxation(cls, *args, **kwargs) -> RelaxMaker:
        """Create a full relaxation maker."""
        return cls(
            input_set_generator=MagneticRelaxSetGenerator(
                *args, relax_cell=True, **kwargs
            )
        )
