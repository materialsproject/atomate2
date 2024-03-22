"""A definition of a MSON document representing an FHI-aims task."""

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
from pymatgen.entries.computed_entries import ComputedEntry

from atomate2.aims.schemas.calculation import AimsObject, Calculation, TaskState
from atomate2.aims.utils import datetime_str

_T = TypeVar("_T", bound="AimsTaskDoc")
_VOLUMETRIC_FILES = ("total_density", "spin_density", "eigenstate_density")
logger = logging.getLogger(__name__)


class AnalysisDoc(BaseModel):
    """Calculation relaxation summary.

    Parameters
    ----------
    delta_volume: float
        Absolute change in volume
    delta_volume_as_percent: float
        Percentage change in volume
    max_force: float
        Maximum force on the atoms
    errors: List[str]
        Errors from the FHI-aims output
    """

    delta_volume: Optional[float] = Field(None, description="Absolute change in volume")
    delta_volume_as_percent: Optional[float] = Field(
        None, description="Percentage change in volume"
    )
    max_force: Optional[float] = Field(None, description="Maximum force on the atoms")
    errors: Optional[list[str]] = Field(
        None, description="Errors from the FHI-aims output"
    )

    @classmethod
    def from_aims_calc_docs(cls, calc_docs: list[Calculation]) -> AnalysisDoc:
        """Create analysis summary from FHI-aims calculation documents.

        Parameters
        ----------
        calc_docs: List[.Calculation]
            FHI-aims calculation documents.

        Returns
        -------
        .AnalysisDoc
            Summary object
        """
        delta_vol = None
        percent_delta_vol = None

        final_calc = calc_docs[-1]
        max_force = None
        if final_calc.has_aims_completed == TaskState.SUCCESS:
            max_force = _get_max_force(final_calc)

        return cls(
            delta_volume=delta_vol,
            delta_volume_as_percent=percent_delta_vol,
            max_force=max_force,
            errors=[],
        )


class Species(BaseModel):
    """A representation of the most important information about each type of species.

    Parameters
    ----------
    element: str
        Element assigned to this atom kind
    species_defaults: str
        Basis set for this atom kind
    """

    element: str = Field(None, description="Element assigned to this atom kind")
    species_defaults: str = Field(None, description="Basis set for this atom kind")


class SpeciesSummary(BaseModel):
    """A summary of species defaults.

    Parameters
    ----------
    species_defaults: Dict[str, .Species]
        Dictionary mapping atomic kind labels to their info
    """

    species_defaults: dict[str, Species] = Field(
        None, description="Dictionary mapping atomic kind labels to their info"
    )

    @classmethod
    def from_species_info(
        cls, species_info: dict[str, dict[str, Any]]
    ) -> SpeciesSummary:
        """Initialize from the atomic_kind_info dictionary.

        Parameters
        ----------
        species_info: Dict[str, Dict[str, Any]]
            The information for the basis set for the calculation

        Returns
        -------
        The SpeciesSummary
        """
        dct: dict[str, dict[str, Any]] = {"species_defaults": {}}
        for kind, info in species_info.items():
            dct["species_defaults"][kind] = {
                "element": info["element"],
                "species_defaults": info["species_defaults"],
            }
        return cls(**dct)


class InputDoc(BaseModel):
    """Summary of inputs for an FHI-aims calculation.

    Parameters
    ----------
    structure: Structure or Molecule
        The input pymatgen Structure or Molecule of the system
    species_info: .SpeciesSummary
        Summary of the species defaults used for each atom kind
    xc: str
        Exchange-correlation functional used if not the default
    """

    structure: Union[Structure, Molecule] = Field(
        None, description="The input structure object"
    )
    species_info: SpeciesSummary = Field(
        None, description="Summary of the species defaults used for each atom kind"
    )
    xc: str = Field(
        None, description="Exchange-correlation functional used if not the default"
    )

    @classmethod
    def from_aims_calc_doc(cls, calc_doc: Calculation) -> InputDoc:
        """Create calculation input summary from a calculation document.

        Parameters
        ----------
        calc_doc: .Calculation
            An FHI-aims calculation document.

        Returns
        -------
        .InputDoc
            A summary of the input structure and parameters.
        """
        summary = SpeciesSummary.from_species_info(calc_doc.input.species_info)

        return cls(
            structure=calc_doc.input.structure,
            atomic_kind_info=summary,
            xc=str(calc_doc.run_type),
        )


class OutputDoc(BaseModel):
    """Summary of the outputs for an FHI-aims calculation.

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
    def from_aims_calc_doc(cls, calc_doc: Calculation) -> OutputDoc:
        """Create a summary from an aims CalculationDocument.

        Parameters
        ----------
        calc_doc: .Calculation
            An FHI-aims calculation document.

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


class ConvergenceSummary(BaseModel):
    """Summary of the outputs for an FHI-aims convergence calculation.

    Parameters
    ----------
    structure: Structure or Molecule
        The output structure object
    converged: bool
        Is convergence achieved?
    convergence_criterion_name: str
        The output name of the convergence criterion
    convergence_field_name: str
        The name of the input setting to study convergence against
    convergence_criterion_value: float
        The output value of the convergence criterion
    convergence_field_value: Any
        The last value of the input setting to study convergence against
    asked_epsilon: float
        The difference in the values for the convergence criteria that was asked for
    actual_epsilon: float
        The actual difference in the convergence criteria values
    """

    structure: Union[Structure, Molecule] = Field(
        None, description="The pymatgen object of the output structure"
    )
    converged: bool = Field(None, description="Is convergence achieved?")

    convergence_criterion_name: str = Field(
        None, description="The output name of the convergence criterion"
    )
    convergence_field_name: str = Field(
        None, description="The name of the input setting to study convergence against"
    )
    convergence_criterion_value: float = Field(
        None, description="The output value of the convergence criterion"
    )
    convergence_field_value: Any = Field(
        None,
        description="The last value of the input setting to study convergence against",
    )
    asked_epsilon: Optional[float] = Field(
        None,
        description="The difference in the values for the convergence criteria that was"
        " asked for",
    )
    actual_epsilon: float = Field(
        None,
        description="The actual difference in the convergence criteria values",
    )

    @classmethod
    def from_aims_calc_doc(cls, calc_doc: Calculation) -> ConvergenceSummary:
        """Create a summary from an FHI-aims calculation document.

        Parameters
        ----------
        calc_doc: .Calculation
            An FHI-aims calculation document.

        Returns
        -------
        .ConvergenceSummary
            The summary for convergence runs.
        """
        from atomate2.aims.jobs.convergence import CONVERGENCE_FILE_NAME

        job_dir = calc_doc.dir_name.split(":")[-1]

        convergence_file = Path(job_dir) / CONVERGENCE_FILE_NAME
        if not convergence_file.exists():
            raise ValueError(
                f"Did not find the convergence json file {CONVERGENCE_FILE_NAME}"
                " in {calc_doc.dir_name}"
            )

        with open(convergence_file) as f:
            convergence_data = json.load(f)

        return cls(
            structure=calc_doc.output.structure,
            converged=convergence_data["converged"],
            convergence_criterion_name=convergence_data["criterion_name"],
            convergence_field_name=convergence_data["convergence_field_name"],
            convergence_criterion_value=convergence_data["criterion_values"][-1],
            convergence_field_value=convergence_data["convergence_field_values"][-1],
            asked_epsilon=None,
            actual_epsilon=abs(
                convergence_data["criterion_values"][-2]
                - convergence_data["criterion_values"][-1]
            ),
        )

    @classmethod
    def from_data(
        cls, structure: Structure | Molecule, convergence_data: dict[str, Any]
    ) -> ConvergenceSummary:
        """Create a summary from the convergence data dictionary.

        Parameters
        ----------
        structure : Structure | Molecule
            a Pymatgen structure object

        convergence_data: dict[str, Any]
            A dictionary containing convergence data for the run.

        Returns
        -------
        .ConvergenceSummary
            The summary for convergence runs.
        """
        return cls(
            structure=structure,
            converged=convergence_data["converged"],
            convergence_criterion_name=convergence_data["criterion_name"],
            convergence_field_name=convergence_data["convergence_field_name"],
            convergence_criterion_value=convergence_data["criterion_values"][-1],
            convergence_field_value=convergence_data["convergence_field_values"][-1],
            asked_epsilon=convergence_data["epsilon"],
            actual_epsilon=abs(
                convergence_data["criterion_values"][-2]
                - convergence_data["criterion_values"][-1]
            ),
        )


class AimsTaskDoc(StructureMetadata, MoleculeMetadata):
    """Definition of FHI-aims task document.

    Parameters
    ----------
    dir_name: str
        The directory for this FHI-aims task
    last_updated: str
        Timestamp for this task document was last updated
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
    included_objects: List[.AimsObject]
        List of FHI-aims objects included with this task document
    aims_objects: Dict[.AimsObject, Any]
        FHI-aims objects associated with this task
    entry: ComputedEntry
        The ComputedEntry from the task doc
    analysis: .AnalysisDoc
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
        The inputs and outputs for all FHI-aims runs in this task.
    transformations: Dict[str, Any]
        Information on the structural transformations, parsed from a
        transformations.json file
    custodian: Any
        Information on the custodian settings used to run this
        calculation, parsed from a custodian.json file
    additional_json: Dict[str, Any]
        Additional json loaded from the calculation directory
    """

    dir_name: str = Field(None, description="The directory for this FHI-aims task")
    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp for this task document was last updated",
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
    included_objects: Optional[list[AimsObject]] = Field(
        None, description="List of FHI-aims objects included with this task document"
    )
    aims_objects: Optional[dict[AimsObject, Any]] = Field(
        None, description="FHI-aims objects associated with this task"
    )
    entry: Optional[ComputedEntry] = Field(
        None, description="The ComputedEntry from the task doc"
    )
    analysis: AnalysisDoc = Field(
        None, description="Summary of structural relaxation and forces"
    )
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
        None, description="The inputs and outputs for all FHI-aims runs in this task."
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
        volumetric_files: Sequence[str] = _VOLUMETRIC_FILES,
        additional_fields: dict[str, Any] = None,
        **aims_calculation_kwargs,
    ) -> AimsTaskDoc:
        """Create a task document from a directory containing FHi-aims files.

        Parameters
        ----------
        dir_name: Path or str
            The path to the folder containing the calculation outputs.
        volumetric_files: Sequence[str]
            A volumetric files to search for.
        additional_fields: Dict[str, Any]
            Dictionary of additional fields to add to output document.
        **aims_calculation_kwargs
            Additional parsing options that will be passed to the
            :obj:`.Calculation.from_aims_files` function.

        Returns
        -------
        .AimsTaskDoc
            A task document for the calculation.
        """
        logger.info(f"Getting task doc in: {dir_name}")

        if additional_fields is None:
            additional_fields = {}

        dir_name = Path(dir_name)
        task_files = _find_aims_files(dir_name, volumetric_files=volumetric_files)

        if len(task_files) == 0:
            raise FileNotFoundError("No FHI-aims files found!")

        calcs_reversed = []
        all_aims_objects = []
        for task_name, files in task_files.items():
            calc_doc, aims_objects = Calculation.from_aims_files(
                dir_name, task_name, **files, **aims_calculation_kwargs
            )
            calcs_reversed.append(calc_doc)
            all_aims_objects.append(aims_objects)

        analysis = AnalysisDoc.from_aims_calc_docs(calcs_reversed)
        tags = additional_fields.get("tags")

        dir_name = get_uri(dir_name)  # convert to full uri path

        # only store objects from last calculation
        # TODO: make this an option
        aims_objects = all_aims_objects[-1]
        included_objects = None
        if aims_objects:
            included_objects = list(aims_objects.keys())

        # rewrite the original structure save!

        data = {
            "structure": calcs_reversed[-1].output.structure,
            "meta_structure": calcs_reversed[-1].output.structure,
            "dir_name": dir_name,
            "calcs_reversed": calcs_reversed,
            "analysis": analysis,
            "tags": tags,
            "completed_at": calcs_reversed[-1].completed_at,
            "output": OutputDoc.from_aims_calc_doc(calcs_reversed[-1]),
            "state": _get_state(calcs_reversed, analysis),
            "entry": cls.get_entry(calcs_reversed),
            "aims_objects": aims_objects,
            "included_objects": included_objects,
        }
        doc = cls(**data)
        return doc.model_copy(update=additional_fields, deep=True)

    @staticmethod
    def get_entry(
        calc_docs: list[Calculation], job_id: Optional[str] = None
    ) -> ComputedEntry:
        """Get a computed entry from a list of FHI-aims calculation documents.

        Parameters
        ----------
        calc_docs: List[.Calculation]
            A list of FHI-aims calculation documents.
        job_id: Optional[str]
            The job identifier.

        Returns
        -------
        ComputedEntry
            A computed entry.
        """
        entry_dict = {
            "correction": 0.0,
            "entry_id": job_id,
            "composition": calc_docs[-1].output.structure.formula,
            "energy": calc_docs[-1].output.energy,
            "parameters": {
                # Required to be compatible with MontyEncoder for the ComputedEntry
                # "run_type": str(calc_docs[-1].run_type),
                "run_type": "AIMS run"
            },
            "data": {
                "last_updated": datetime_str(),
            },
        }
        return ComputedEntry.from_dict(entry_dict)


def _find_aims_files(
    path: Path | str,
    volumetric_files: Sequence[str] = _VOLUMETRIC_FILES,
) -> dict[str, Any]:
    """Find FHI-aims files in a directory.

    Only files in folders with names matching a task name (or alternatively files
    with the task name as an extension, e.g., aims.out) will be returned.

    FHI-aims files in the current directory will be given the task name "standard".

    Parameters
    ----------
    path: str or Path
        Path to a directory to search.
    volumetric_files: Sequence[str]
        Volumetric files to search for.

    Returns
    -------
    dict[str, Any]
        The filenames of the calculation outputs for each FHI-aims task,
        given as a ordered dictionary of::

            {
                task_name: {
                    "aims_output_file": aims_output_filename,
                    "volumetric_files": [v_hartree file, e_density file, etc],
    """
    task_names = ["precondition"] + [f"relax{i}" for i in range(9)]
    path = Path(path)
    task_files = {}

    def _get_task_files(
        files: list[Path], suffix: str = ""
    ) -> dict[str, list[str | Path] | Path | str]:
        aims_files: dict[str, list[str | Path] | Path | str] = {}
        vol_files: list[str | Path] = []
        for file in files:
            # Here we make assumptions about the output file naming
            if file.match(f"*aims.out{suffix}*"):
                aims_files["aims_output_file"] = Path(file).name
        for vol in volumetric_files:
            _files = [f.name for f in files if f.match(f"*{vol}*cube{suffix}*")]
            if _files:
                vol_files.append(_files[0])

        if len(vol_files) > 0:
            # add volumetric files if some were found or other cp2k files were found
            aims_files["volumetric_files"] = vol_files

        return aims_files

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


def _get_max_force(calc_doc: Calculation) -> Optional[float]:
    """Get max force acting on atoms from a calculation document.

    Parameters
    ----------
    calc_doc: .Calculation
        The calculation doc to get the max force

    Returns
    -------
    float
        The maximum force
    """
    forces = calc_doc.output.forces
    if forces is not None:
        forces = np.array(forces)
        return max(np.linalg.norm(forces, axis=1))
    return None


def _get_state(calc_docs: list[Calculation], analysis: AnalysisDoc) -> TaskState:
    """Get state from calculation documents and relaxation analysis.

    Parameters
    ----------
    calc_docs: List[.Calculations]
        The calculation to get the state from
    analysis: .AnalysisDoc
        The summary of the analysis

    Returns
    -------
    .TaskState
        The status of the calculation
    """
    all_calcs_completed = all(
        c.has_aims_completed == TaskState.SUCCESS for c in calc_docs
    )
    if len(analysis.errors) == 0 and all_calcs_completed:
        return TaskState.SUCCESS  # type: ignore  # noqa: PGH003
    return TaskState.FAILED  # type: ignore  # noqa: PGH003
