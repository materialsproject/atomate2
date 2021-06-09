"""Definition of base VASP job maker."""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

from jobflow import Maker, job

if typing.TYPE_CHECKING:
    from pymatgen.core import Structure

__all__ = ["BaseVaspMaker"]


@dataclass
class BaseVaspMaker(Maker):
    """Base VASP job maker."""

    name: str = "base vasp job"
    input_set: str = None
    input_set_kwargs: dict = field(default_factory=dict)
    copy_vasp_kwargs: dict = field(default_factory=dict)
    run_vasp_kwargs: dict = field(default_factory=dict)
    vasp_output_kwargs: dict = field(default_factory=dict)

    @job
    def make(self, structure: Structure, prev_vasp_dir=None):
        """Make a VASP job."""
        from jobflow.utils.dict_mods import apply_mod

        from atomate2.vasp.file import copy_vasp_outputs
        from atomate2.vasp.parse import parse_vasp_outputs

        if prev_vasp_dir is not None:
            copy_vasp_outputs(prev_vasp_dir, **self.copy_vasp_kwargs)
        #
        # write_vasp_input_set(
        #     structure,
        #     self.vasp_input_set,
        #     self.vasp_input_set_kwargs,
        #     from_prev=from_prev,
        # )
        # run_vasp(**self.run_vasp_kwargs)

        apply_mod(
            self.vasp_output_kwargs,
            {"_set": {"additional_field->task_label": self.name}},
        )
        response = parse_vasp_outputs(**self.vasp_output_kwargs)
        return response
