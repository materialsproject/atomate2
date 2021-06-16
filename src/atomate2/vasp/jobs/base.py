"""Definition of base VASP job maker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from jobflow import Maker, job
from pymatgen.core import Structure

from atomate2.vasp.schemas.task import TaskDocument

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

    @job(output_schema=TaskDocument)
    def make(self, structure: Structure, prev_vasp_dir: Union[str, Path] = None):
        """Make a VASP job."""
        raise NotImplementedError
