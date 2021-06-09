"""Core jobs for running VASP calculations."""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

from jobflow import Response, job

from atomate2.vasp.drones import VaspDrone
from atomate2.vasp.file import copy_vasp_outputs
from atomate2.vasp.inputs import write_vasp_input_set
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.run import run_vasp, should_stop_children

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
    vasp_drone_kwargs: dict = field(default_factory=dict)
    stop_children_kwargs: dict = field(default_factory=dict)

    @job
    def make(self, structure: Structure, prev_vasp_dir=None):
        """Make a VASP job."""
        # copy previous inputs
        from_prev = prev_vasp_dir is not None
        if prev_vasp_dir is not None:
            copy_vasp_outputs(prev_vasp_dir, **self.copy_vasp_kwargs)

        if "from_prev" not in self.write_vasp_input_set_kwargs:
            self.write_vasp_input_set_kwargs["from_prev"] = from_prev

        # write vasp input files
        write_vasp_input_set(
            structure,
            self.input_set,
            self.input_set_kwargs,
            **self.write_vasp_input_set_kwargs
        )

        # run vasp
        run_vasp(**self.run_vasp_kwargs)

        # parse vasp outputs
        drone = VaspDrone(**self.vasp_drone_kwargs)
        task_doc = drone.assimilate()
        task_doc.task_label = self.name

        # decide whether child jobs should proceed
        stop_children = should_stop_children(task_doc, **self.stop_children_kwargs)

        return Response(
            stop_children=stop_children,
            stored_data={"custodian": task_doc.custodian},
            output=task_doc,
        )
