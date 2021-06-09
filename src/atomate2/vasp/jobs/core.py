"""Core jobs for running VASP calculations."""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

from jobflow import job

from atomate2.vasp.jobs.base import BaseVaspMaker

if typing.TYPE_CHECKING:
    from pymatgen.core.structure import Structure


@dataclass
class RelaxMaker(BaseVaspMaker):
    """Maker to create VASP relaxation jobs."""

    name: str = "structure relaxation"
    input_set: str = "MPRelaxSet"


@dataclass
class StaticMaker(BaseVaspMaker):
    """Maker to create VASP static jobs."""

    name: str = "structure relaxation"
    input_set: str = "MPStaticSet"
    input_set_kwargs: dict = field(default_factory=dict)
    write_vasp_input_set_kwargs: dict = field(default_factory=dict)
    copy_vasp_kwargs: dict = field(default_factory=dict)
    run_vasp_kwargs: dict = field(default_factory=dict)
    parse_vasp_output_kwargs: dict = field(default_factory=dict)

    @job
    def make(self, structure: Structure, prev_vasp_dir=None):
        """Make a VASP job."""
        from jobflow import Response
        from jobflow.utils.dict_mods import apply_mod

        from atomate2.vasp.file import copy_vasp_outputs
        from atomate2.vasp.inputs import write_vasp_input_set
        from atomate2.vasp.parse import parse_vasp_outputs
        from atomate2.vasp.run import run_vasp

        from_prev = False
        if prev_vasp_dir is not None:
            copy_vasp_outputs(prev_vasp_dir, **self.copy_vasp_kwargs)
            from_prev = True

        if "from_prev" not in self.write_vasp_input_set_kwargs:
            self.write_vasp_input_set_kwargs["from_prev"] = from_prev

        write_vasp_input_set(
            structure,
            self.input_set,
            self.input_set_kwargs,
            **self.write_vasp_input_set_kwargs
        )
        custodian_output = run_vasp(**self.run_vasp_kwargs)

        apply_mod(
            self.parse_vasp_output_kwargs,
            {"_set": {"vasp_drone_kwargs->additional_fields->task_label": self.name}},
        )
        task_doc, stop_children = parse_vasp_outputs(**self.parse_vasp_output_kwargs)

        return Response(
            stop_children=stop_children,
            stored_data={"custodian": custodian_output},
            output=task_doc,
        )
