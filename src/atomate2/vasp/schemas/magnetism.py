"""Schemas for magnetic ordering calculations in VASP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from atomate2.common.schemas.magnetism import (
    MagneticOrderingOutput as MagneticOrderingOutputBase,
)
from atomate2.common.schemas.magnetism import (
    MagneticOrderingRelaxation as MagneticOrderingRelaxationBase,
)

if TYPE_CHECKING:
    from emmet.core.tasks import TaskDoc


class MagneticOrderingRelaxation(MagneticOrderingRelaxationBase):
    """
    Defines the relaxation output document for a magnetic ordering calculation.

    The construction of this document is implemented here for VASP. See base class for
    more details.
    """

    @classmethod
    def from_task_document(cls, task_document: TaskDoc, uuid: str | None = None):
        """Construct a MagneticOrderingRelaxation from a task document.

        This does not include the uuid, which must be set separately.
        """
        dir_name = task_document.dir_name
        structure = task_document.structure
        input_structure = task_document.input.structure
        if not input_structure.site_properties.get("magmom"):
            input_structure.add_site_property(  # input struct likely has no magmoms
                "magmom", task_document.input.parameters["MAGMOM"]
            )
        energy = task_document.output.energy

        return cls.from_structures_and_energies(
            input_structure=input_structure,
            output_structure=structure,
            output_energy=energy,
            uuid=uuid,
            dir_name=dir_name,
        )


class MagneticOrderingOutput(MagneticOrderingOutputBase):
    """Defines the static output for a magnetic ordering calculation.

    The construction of this document is implemented here for VASP. See base class for
    more details.
    """

    @classmethod
    def from_task_document(
        cls,
        task_document: TaskDoc,
        uuid: str | None = None,
        relax_output: MagneticOrderingRelaxation | None = None,
    ):
        """Construct a MagneticOrderingOutput from a task document.

        This does not include the uuid, which must be set separately.
        """
        dir_name = task_document.dir_name
        structure = task_document.structure
        input_structure = task_document.input.structure
        if not input_structure.site_properties.get("magmom"):
            input_structure.add_site_property(  # input struct likely has no magmoms
                "magmom", task_document.input.parameters["MAGMOM"]
            )
        energy = task_document.output.energy

        return cls.from_structures_and_energies(
            input_structure=input_structure,
            output_structure=structure,
            output_energy=energy,
            uuid=uuid,
            dir_name=dir_name,
            relax_output=relax_output,
        )
