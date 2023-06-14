"""Jobs used for enumeration, calculation, and analysis of magnetic orderings."""

from __future__ import annotations

import logging
from typing import Literal, Optional, Sequence

from jobflow import Flow, Response, job
from pymatgen.analysis.magnetism import MagneticStructureEnumerator
from pymatgen.core import Element
from pymatgen.core.structure import Structure

from atomate2.vasp.jobs.base import BaseVaspMaker

logger = logging.getLogger(__name__)

__all__ = [
    "enumerate_magnetic_orderings",
    "run_ordering_calculations",
    "analyze_orderings",
]


@job
def enumerate_magnetic_orderings(
    structure: Structure,
    default_magmoms: dict[str, float] | None = None,
    strategies: list[str] | tuple[str, ...] = ("ferromagnetic", "antiferromagnetic"),
    automatic: bool = True,
    truncate_by_symmetry: bool = True,
    transformation_kwargs: dict | None = None,
):

    """ 
    """
    enumerator = MagneticStructureEnumerator(
        structure,
        default_magmoms=default_magmoms,
        strategies=tuple(strategies),  # TODO: this type hint could be changed in pymatgen
        automatic=automatic,
        truncate_by_symmetry=truncate_by_symmetry,
        transformation_kwargs=transformation_kwargs
    )

    return enumerator.ordered_structures


@job
def run_ordering_calculations(
    orderings: list[Structure],
    # prev_vasp_dir: str | Path | None = None,  # TODO: find out how to handle N prev_vasp_dirs
    maker: BaseVaspMaker,
):
    """
    Run magnetic ordering calculations.

    This job will automatically replace itself with calculation 

    Parameters
    ----------
    orderings : List[Structure]
        A list of pymatgen structures.
    maker : .BaseVaspMaker
        A VaspMaker to use to calculate the energies of the orderings.
    """

    jobs = []
    outputs = []
    for i, ordering in enumerate(orderings):

        # TODO: think about additional data here from enumerator? c.f. elastic:
        # elastic_relax_maker.write_additional_data["transformations:json"] = ts

        # create the job
        job = maker.make(
            ordering#, prev_vasp_dir=prev_vasp_dir
        )
        job.append_name(f" {i + 1}/{len(orderings)}")
        jobs.append(job)

        # extract the outputs we want
        output = {
            # TODO: check output format here
            "energy": job.output.output.energy,
            "uuid": job.output.uuid,
            "job_dir": job.output.dir_name,
        }

        outputs.append(output)

    flow = Flow(jobs, outputs)
    return Response(replace=flow)

@job#(output_schema=...)
def analyze_orderings(*args):
    raise NotImplementedError
