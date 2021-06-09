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
    write_vasp_input_set_kwargs: dict = field(default_factory=dict)
    copy_vasp_kwargs: dict = field(default_factory=dict)
    run_vasp_kwargs: dict = field(default_factory=dict)
    vasp_drone_kwargs: dict = field(default_factory=dict)
    stop_children_kwargs: dict = field(default_factory=dict)

    @job
    def make(self, structure: Structure, prev_vasp_dir=None):
        """Make a VASP job."""
        raise NotImplementedError
