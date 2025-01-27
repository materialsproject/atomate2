"""Run MACE on randomly packed benchmarking structures."""

import io
import json
from pathlib import Path

import numpy as np
import openmm
import openmm.unit as omm_unit
from atoms2.comp.md.atomate2.utils import structure_to_topology
from emmet.core.openmm import OpenMMInterchange, OpenMMTaskDocument
from emmet.core.vasp.task_valid import TaskState
from jobflow import Response
from mace.calculators.foundations_models import download_mace_mp_checkpoint
from monty.json import MontyEncoder
from openmm import Context, XmlSerializer
from openmm.app.pdbfile import PDBFile
from pymatgen.core import Structure

from atomate2.openmm.jobs.base import openmm_job
from atomate2.openmm.mace_force import MacePotential


@openmm_job
def generate_mace_interchange(
    structure: Structure,
    ff_path: str | Path | None = None,
    tags: list[str] | None = None,
) -> Response:
    """Generate an OpenMMInterchange object with the MACE force-field.

    Parameters
    ----------
    structure : Structure
        The structure to simulate.
    ff_path : str | Path, optional
        The path to the MACE force-field. Must be accessible where the job is run.
        Defaults to None.
    tags : list[str], optional
        Tags to add to the task document. Defaults to None.

    Returns
    -------
    Response
        The response containing the OpenMMTaskDocument.
    """
    if not ff_path:
        ff_path = Path(download_mace_mp_checkpoint())

    potential = MacePotential(model_path=ff_path)

    topology = structure_to_topology(structure)
    topology.setPeriodicBoxVectors(structure.lattice.matrix / 10)
    system = potential.create_system(topology)
    integrator = openmm.LangevinIntegrator(
        300 * omm_unit.kelvin, 10.0 / omm_unit.picoseconds, 1.0 * omm_unit.femtosecond
    )
    context = Context(system, integrator)
    context.setPositions(structure.cart_coords / 10)
    state = context.getState(getPositions=True)
    with io.StringIO() as buffer:
        PDBFile.writeFile(topology, np.zeros(shape=(len(structure), 3)), file=buffer)
        buffer.seek(0)
        pdb = buffer.read()

    interchange = OpenMMInterchange(
        system=XmlSerializer.serialize(system),
        state=XmlSerializer.serialize(state),
        topology=pdb,
    )

    interchange_json = interchange.model_dump_json()

    dir_name = Path.cwd()

    task_doc = OpenMMTaskDocument(
        dir_name=str(dir_name),
        state=TaskState.SUCCESS,
        interchange=interchange_json,
        structure=structure,
        force_field=Path(ff_path).stem,
        tags=tags,
    )

    # write out task_doc json to output dir
    with open(dir_name / "taskdoc.json", "w") as file:
        json.dump(task_doc.model_dump(), file, cls=MontyEncoder)

    return Response(output=task_doc)
