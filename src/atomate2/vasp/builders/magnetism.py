"""Module defining VASP magnetic orderings builder."""

from __future__ import annotations

from atomate2.common.builders.magnetism import (
    MagneticOrderingsBuilder as MagneticOrderingsBuilderBase,
)
from atomate2.vasp.schemas.magnetism import (
    MagneticOrderingOutput,
    MagneticOrderingRelaxation,
)


class MagneticOrderingsBuilder(MagneticOrderingsBuilderBase):
    """Builder to analyze the results of magnetic orderings calculations.

    This is implemented for VASP by defining methods below.
    """

    def _build_relax_output(self, relax_task, uuid=None):
        return MagneticOrderingRelaxation.from_task_document(relax_task, uuid=uuid)

    def _build_static_output(self, static_task, uuid=None, relax_output=None):
        return MagneticOrderingOutput.from_task_document(
            static_task, uuid=uuid, relax_output=relax_output
        )

    @property
    def _dft_code_query(self):
        return {"output.orig_inputs.incar": {"$exists": True}}  # ensure VASP calcs only
