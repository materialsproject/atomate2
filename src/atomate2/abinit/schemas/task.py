"""A definition of a MSON document representing an Abinit task."""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import numpy as np
from emmet.core.math import Matrix3D, Vector3D
from emmet.core.structure import MoleculeMetadata, StructureMetadata
from emmet.core.tasks import get_uri
from pydantic import BaseModel, Field
from pymatgen.core import Molecule, Structure
#from pymatgen.entries.computed_entries import ComputedEntry

#from atomate2.abinit.schemas.calculation import AbinitObject, Calculation, TaskState
from atomate2.abinit.schemas.calculation import Calculation
from atomate2.abinit.utils import datetime_str

_T = TypeVar("_T", bound="AbinitTaskDoc")
#_VOLUMETRIC_FILES = ("total_density", "spin_density", "eigenstate_density")
logger = logging.getLogger(__name__)

class OutputDoc(BaseModel):
    """Summary of the outputs for an Abinit calculation.

    Parameters
    ----------
    structure: Structure or Molecule
        The final pymatgen Structure or Molecule of the final system
    trajectory: List[Structure or Molecule]
        The trajectory of output structures
    energy: float
        The final total DFT energy for the last calculation
    energy_per_atom: float
        The final DFT energy per atom for the last calculation
    bandgap: float
        The DFT bandgap for the last calculation
    cbm: float
        CBM for this calculation
    vbm: float
        VBM for this calculation
    forces: List[Vector3D]
        Forces on atoms from the last calculation
    stress: Matrix3D
        Stress on the unit cell from the last calculation
    all_forces: List[List[Vector3D]]
        Forces on atoms from all calculations.
    """

    structure: Union[Structure, Molecule] = Field(
        None, description="The output structure object"
    )
    trajectory: Sequence[Union[Structure, Molecule]] = Field(
        None, description="The trajectory of output structures"
    )
    energy: float = Field(
        None, description="The final total DFT energy for the last calculation"
    )
    energy_per_atom: float = Field(
        None, description="The final DFT energy per atom for the last calculation"
    )
    bandgap: Optional[float] = Field(
        None, description="The DFT bandgap for the last calculation"
    )
    cbm: Optional[float] = Field(None, description="CBM for this calculation")
    vbm: Optional[float] = Field(None, description="VBM for this calculation")
    forces: Optional[list[Vector3D]] = Field(
        None, description="Forces on atoms from the last calculation"
    )
    stress: Optional[Matrix3D] = Field(
        None, description="Stress on the unit cell from the last calculation"
    )
    all_forces: Optional[list[list[Vector3D]]] = Field(
        None, description="Forces on atoms from all calculations."
    )

    @classmethod
    def from_abinit_calc_doc(cls, calc_doc: Calculation) -> OutputDoc:
        """Create a summary from an abinit CalculationDocument.

        Parameters
        ----------
        calc_doc: .Calculation
            An Abinit calculation document.

        Returns
        -------
        .OutputDoc
            The calculation output summary.
        """
        return cls(
            structure=calc_doc.output.structure,
            energy=calc_doc.output.energy,
            energy_per_atom=calc_doc.output.energy_per_atom,
            bandgap=calc_doc.output.bandgap,
            cbm=calc_doc.output.cbm,
            vbm=calc_doc.output.vbm,
            forces=calc_doc.output.forces,
            stress=calc_doc.output.stress,
            all_forces=calc_doc.output.all_forces,
            trajectory=calc_doc.output.atomic_steps,
        )

class AbinitTaskDoc(StructureMetadata, MoleculeMetadata):
    """Definition of Abinit task document.

    Parameters
    ----------
    dir_name: str
        The directory for this Abinit task
    last_updated: str
        Timestamp for when this task document was last updated
    completed_at: str
        Timestamp for when this task was completed
    input: .InputDoc
        The input to the first calculation
    output: .OutputDoc
        The output of the final calculation
    structure: Structure or Molecule
        Final output structure from the task
    state: .TaskState
        State of this task
    included_objects: List[.AbinitObject]
        List of Abinit objects included with this task document
    abinit_objects: Dict[.AbinitObject, Any]
        Abinit objects associated with this task
    #entry: ComputedEntry
        #The ComputedEntry from the task doc
    #analysis: .AnalysisDoc
        Summary of structural relaxation and forces
    task_label: str
        A description of the task
    tags: List[str]
        Metadata tags for this task document
    author: str
        Author extracted from transformations
    icsd_id: str
        International crystal structure database id of the structure
    calcs_reversed: List[.Calculation]
        The inputs and outputs for all Abinit runs in this task.
    transformations: Dict[str, Any]
        Information on the structural transformations, parsed from a
        transformations.json file
    custodian: Any
        Information on the custodian settings used to run this
        calculation, parsed from a custodian.json file
    additional_json: Dict[str, Any]
        Additional json loaded from the calculation directory
    """

    dir_name: str = Field(None, description="The directory for this Abinit task")
    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp for when this task document was last updated",
    )
    completed_at: str = Field(
        None, description="Timestamp for when this task was completed"
    )
    input: Optional[InputDoc] = Field(
        None, description="The input to the first calculation"
    )
    output: OutputDoc = Field(None, description="The output of the final calculation")
    structure: Union[Structure, Molecule] = Field(
        None, description="Final output atoms from the task"
    )
    state: TaskState = Field(None, description="State of this task")
    included_objects: Optional[list[AbinitObject]] = Field(
        None, description="List of Abinit objects included with this task document"
    )
    abinit_objects: Optional[dict[AbinitObject, Any]] = Field(
        None, description="Abinit objects associated with this task"
    )
    #entry: Optional[ComputedEntry] = Field(
    #    None, description="The ComputedEntry from the task doc"
    #)
    #analysis: AnalysisDoc = Field(
    #    None, description="Summary of structural relaxation and forces"
    #)
    task_label: str = Field(None, description="A description of the task")
    tags: Optional[list[str]] = Field(
        None, description="Metadata tags for this task document"
    )
    author: Optional[str] = Field(
        None, description="Author extracted from transformations"
    )
    icsd_id: Optional[str] = Field(
        None, description="International crystal structure database id of the structure"
    )
    calcs_reversed: Optional[list[Calculation]] = Field(
        None, description="The inputs and outputs for all Abinit runs in this task."
    )
    transformations: Optional[dict[str, Any]] = Field(
        None,
        description="Information on the structural transformations, parsed from a "
        "transformations.json file",
    )
    custodian: Any = Field(
        None,
        description="Information on the custodian settings used to run this "
        "calculation, parsed from a custodian.json file",
    )
    additional_json: Optional[dict[str, Any]] = Field(
        None, description="Additional json loaded from the calculation directory"
    )

    @classmethod
    def from_directory(
        cls: type[_T],
        dir_name: Path | str,
        #volumetric_files: Sequence[str] = _VOLUMETRIC_FILES,
        additional_fields: dict[str, Any] = None,
        **abinit_calculation_kwargs,
    ):
        """Create a task document from a directory containing Abinit files.

        Parameters
        ----------
        dir_name: Path or str
            The path to the folder containing the calculation outputs.
        #volumetric_files: Sequence[str]
            #A volumetric files to search for.
        additional_fields: Dict[str, Any]
            Dictionary of additional fields to add to output document.
        **abinit_calculation_kwargs
            Additional parsing options that will be passed to the
            :obj:`.Calculation.from_abinit_files` function.

        Returns
        -------
        .AbinitTaskDoc
            A task document for the calculation.
        """
        logger.info(f"Getting task doc in: {dir_name}")

        if additional_fields is None:
            additional_fields = {}

        dir_name = Path(dir_name)
        task_files = _find_abinit_files(dir_name)#, volumetric_files=volumetric_files)

        if len(task_files) == 0:
            raise FileNotFoundError("No Abinit files found!")

        calcs_reversed = []
        all_abinit_objects = []
        for task_name, files in task_files.items():
            calc_doc, abinit_objects = Calculation.from_abinit_files(
                dir_name, task_name, **files, **abinit_calculation_kwargs
            )
            calcs_reversed.append(calc_doc)
            all_abinit_objects.append(abinit_objects)

        #analysis = AnalysisDoc.from_abinit_calc_docs(calcs_reversed)
        tags = additional_fields.get("tags")

        dir_name = get_uri(dir_name)  # convert to full uri path

        # only store objects from last calculation
        # TODO: make this an option
        abinit_objects = all_abinit_objects[-1]
        included_objects = None
        if abinit_objects:
            included_objects = list(abinit_objects.keys())

        # rewrite the original structure save!

        data = {
            "structure": calcs_reversed[-1].output.structure,
            "meta_structure": calcs_reversed[-1].output.structure,
            "dir_name": dir_name,
            "calcs_reversed": calcs_reversed,
            #"analysis": analysis,
            "tags": tags,
            "completed_at": calcs_reversed[-1].completed_at,
            "output": OutputDoc.from_abinit_calc_doc(calcs_reversed[-1]),
            # "state": _get_state(calcs_reversed),#, analysis),
            #"entry": cls.get_entry(calcs_reversed),
            "abinit_objects": abinit_objects,
            "included_objects": included_objects,
        }
        doc = cls(**data)
        return doc.model_copy(update=additional_fields, deep=True)

    #@staticmethod
    #def get_entry(
    #    calc_docs: list[Calculation], job_id: Optional[str] = None
    #) -> ComputedEntry:
    #    """Get a computed entry from a list of Abinit calculation documents.

    #    Parameters
    #    ----------
    #    calc_docs: List[.Calculation]
    #        A list of Abinit calculation documents.
    #    job_id: Optional[str]
    #        The job identifier.

    #    Returns
    #    -------
    #    ComputedEntry
    #        A computed entry.
    #    """
    #    entry_dict = {
    #        "correction": 0.0,
    #        "entry_id": job_id,
    #        "composition": calc_docs[-1].output.structure.formula,
    #        "energy": calc_docs[-1].output.energy,
    #        "parameters": {
    #            # Required to be compatible with MontyEncoder for the ComputedEntry
    #            # "run_type": str(calc_docs[-1].run_type),
    #            "run_type": "abinit run"
    #        },
    #        "data": {
    #            "last_updated": datetime_str(),
    #        },
    #    }
    #    return ComputedEntry.from_dict(entry_dict)



def _find_abinit_files(
    path: Path | str,
    #volumetric_files: Sequence[str] = _VOLUMETRIC_FILES,
) -> dict[str, Any]:
    """Find Abinit files in a directory.

    Only files in folders with names matching a task name (or alternatively files
    with the task name as an extension, e.g., abinit.out) will be returned.

    Abinit files in the current directory will be given the task name "standard".

    Parameters
    ----------
    path: str or Path
        Path to a directory to search.
    #volumetric_files: Sequence[str]
        #Volumetric files to search for.

    Returns
    -------
    dict[str, Any]
        The filenames of the calculation outputs for each Abinit task,
        given as a ordered dictionary of::

            {
                task_name: {
                    "abinit_output_file": abinit_output_filename,
                    #"volumetric_files": [v_hartree file, e_density file, etc],
    """
    task_names = ["precondition"] + [f"relax{i}" for i in range(9)]
    path = Path(path)
    task_files = {}

    def _get_task_files(files, suffix=""):
        abinit_files = {}
        #vol_files = []
        for file in files:
            # Here we make assumptions about the output file naming
            if file.match(f"*outdata/out_GSR{suffix}*"):
                # abinit_files["abinit_output_file"] = Path(file).name
                abinit_files["abinit_output_file"] = Path(file).relative_to(path)
        #for vol in volumetric_files:
        #    _files = [f.name for f in files if f.match(f"*{vol}*cube{suffix}*")]
        #    if _files:
        #        vol_files.append(_files[0])

        #if len(vol_files) > 0:
        #    # add volumetric files if some were found or other cp2k files were found
        #    abinit_files["volumetric_files"] = vol_files

        return abinit_files

    for task_name in task_names:
        subfolder_match = list(path.glob(f"{task_name}/*"))
        suffix_match = list(path.glob(f"*.{task_name}*"))
        if len(subfolder_match) > 0:
            # subfolder match
            task_files[task_name] = _get_task_files(subfolder_match)
        elif len(suffix_match) > 0:
            # try extension schema
            task_files[task_name] = _get_task_files(
                suffix_match, suffix=f".{task_name}"
            )

    if len(task_files) == 0:
        # get any matching file from the root folder
        standard_files = _get_task_files(list(path.glob("*")))
        if len(standard_files) > 0:
            task_files["standard"] = standard_files

    return task_files
