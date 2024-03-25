"""Module defining VASP magnetic orderings builder."""

from __future__ import annotations

from atomate2.common.builders.magnetism import (
    MagneticOrderingsBuilder as MagneticOrderingsBuilderBase,
)
from atomate2.vasp.schemas.magnetism import MagneticOrderingsDocument


class MagneticOrderingsBuilder(MagneticOrderingsBuilderBase):
    """Builder to analyze the results of magnetic orderings calculations.

    This is implemented for VASP by defining the methods below.
    """

    @staticmethod
    def _build_doc_fn(tasks):
        return MagneticOrderingsDocument.from_tasks(tasks)

    @property
    def _dft_code_query(self):
        return {"output.orig_inputs.incar": {"$exists": True}}  # ensure VASP calcs only
