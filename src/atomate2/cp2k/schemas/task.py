"""Core definition of a CP2K task document."""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import numpy as np
from emmet.core.math import Matrix3D, Vector3D
from emmet.core.structure import MoleculeMetadata, StructureMetadata
from pydantic import BaseModel, Field
from pymatgen.core.structure import Molecule, Structure
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.cp2k.inputs import Cp2kInput
from pymatgen.io.cp2k.utils import natural_keys

from atomate2 import SETTINGS, __version__
from atomate2.common.utils import (
    parse_additional_json,
    parse_custodian,
    parse_transformations,
)
from atomate2.cp2k.schemas.calculation import (
    Calculation,
    Cp2kObject,
    RunStatistics,
    Status,
)
from atomate2.utils.datetime import datetime_str
from atomate2.utils.path import get_uri

logger = logging.getLogger(__name__)
_T = TypeVar("_T", bound="TaskDocument")
_VOLUMETRIC_FILES = ("v_hartree", "ELECTRON_DENSITY", "SPIN_DENSITY")


class AnalysisSummary(BaseModel):
    """Calculation relaxation summary."""

    delta_volume: float = Field(None, description="Absolute change in volume")
    delta_volume_as_percent: float = Field(
        None, description="Percentage change in volume"
    )
    max_force: float = Field(None, description="Maximum force on the atoms")
    warnings: list[str] = Field(None, description="Warnings from the VASP drone")
    errors: list[str] = Field(None, description="Errors from the VASP drone")

    @classmethod
    def from_cp2k_calc_docs(cls, calc_docs: list[Calculation]) -> "AnalysisSummary":
        """
        Create analysis summary from CP2K calculation documents.

        Parameters
        ----------
        calc_docs
            VASP calculation documents.

        Returns
        -------
        AnalysisSummary
            The relaxation analysis.
        """
        from atomate2.cp2k.schemas.calculation import Status

        warnings = []
        errors = []

        if isinstance(calc_docs[0].input.structure, Structure):
            initial_vol = calc_docs[0].input.structure.lattice.volume
            final_vol = calc_docs[-1].output.structure.lattice.volume
            delta_vol = final_vol - initial_vol
            percent_delta_vol = 100 * delta_vol / initial_vol

            if abs(percent_delta_vol) > SETTINGS.CP2K_VOLUME_CHANGE_WARNING_TOL * 100:
                warnings.append(
                    f"Volume change > {SETTINGS.CP2K_VOLUME_CHANGE_WARNING_TOL * 100}%"
                )
        else:
            delta_vol = None
            percent_delta_vol = None

        final_calc = calc_docs[-1]
        max_force = None
        if final_calc.has_cp2k_completed == Status.SUCCESS:
            # max force and valid structure checks
            structure = final_calc.output.structure
            max_force = _get_max_force(final_calc)
            if not structure.is_valid():
                errors.append("Bad structure (atoms are too close!)")

        return cls(
            delta_volume=delta_vol,
            delta_volume_as_percent=percent_delta_vol,
            max_force=max_force,
            warnings=warnings,
            errors=errors,
        )


class AtomicKind(BaseModel):
    """A representation of the most important information about each type of species."""

    element: str = Field(None, description="Element assigned to this atom kind")
    basis: str = Field(None, description="Basis set for this atom kind")
    potential: str = Field(
        None, description="Name of pseudopotential for this atom kind"
    )
    auxiliary_basis: Optional[str] = Field(
        None, description="Auxiliary basis for this (if any) for this atom kind"
    )
    ghost: bool = Field(None, description="Whether this atom kind is a ghost")


class AtomicKindSummary(BaseModel):
    """A summary of pseudo-potential type and functional."""

    atomic_kinds: dict[str, AtomicKind] = Field(
        None, description="dictionary mapping atomic kind labels to their info"
    )

    @classmethod
    def from_atomic_kind_info(cls, atomic_kind_info: dict) -> "AtomicKindSummary":
        """Initialize from the atomic_kind_info dictionary."""
        d: dict[str, dict[str, Any]] = {"atomic_kinds": {}}
        for kind, info in atomic_kind_info.items():
            d["atomic_kinds"][kind] = {
                "element": info["element"],
                "basis": info["orbital_basis_set"],
                "potential": info["pseudo_potential"],
                "auxiliary_basis": info["auxiliary_basis_set"][0]
                if info["auxiliary_basis_set"]
                else None,
                "ghost": info["pseudo_potential"] == "NONE",
            }
        return cls(**d)


class InputSummary(BaseModel):
    """Summary of inputs for a CP2K calculation."""

    structure: Union[Structure, Molecule] = Field(
        None, description="The input structure object"
    )

    atomic_kind_info: AtomicKindSummary = Field(
        None, description="Summary of the potential and basis used for each atom kind"
    )
    xc: str = Field(
        None, description="Exchange-correlation functional used if not the default"
    )

    @classmethod
    def from_cp2k_calc_doc(cls, calc_doc: Calculation) -> "InputSummary":
        """
        Create calculation input summary from a calculation document.

        Parameters
        ----------
        calc_doc
            A CP2K calculation document.

        Returns
        -------
        InputSummary
            A summary of the input structure and parameters.
        """
        summary = AtomicKindSummary.from_atomic_kind_info(
            calc_doc.input.atomic_kind_info
        )

        return cls(
            structure=calc_doc.input.structure,
            atomic_kind_info=summary,
            xc=str(calc_doc.run_type),
        )


class OutputSummary(BaseModel):
    """Summary of the outputs for a CP2K calculation."""

    structure: Union[Structure, Molecule] = Field(
        None, description="The output structure object"
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
    forces: list[Vector3D] = Field(
        None, description="Forces on atoms from the last calculation"
    )
    stress: Optional[Matrix3D] = Field(
        None, description="Stress on the unit cell from the last calculation"
    )

    @classmethod
    def from_cp2k_calc_doc(cls, calc_doc: Calculation) -> "OutputSummary":
        """
        Create a summary of CP2K calculation outputs from a CP2K calculation document.

        Parameters
        ----------
        calc_doc
            A CP2K calculation document.

        Returns
        -------
        OutputSummary
            The calculation output summary.
        """
        if calc_doc.output.ionic_steps:
            forces = calc_doc.output.ionic_steps[-1].get("forces")
            stress = calc_doc.output.ionic_steps[-1].get("stress")
        else:
            forces = None
            stress = None
        return cls(
            structure=calc_doc.output.structure,
            energy=calc_doc.output.energy,
            energy_per_atom=calc_doc.output.energy_per_atom,
            bandgap=calc_doc.output.bandgap,
            cbm=calc_doc.output.cbm,
            vbm=calc_doc.output.vbm,
            forces=forces,
            stress=stress,
        )


class TaskDocument(StructureMetadata, MoleculeMetadata):
    """Definition of CP2K task document."""

    dir_name: Optional[str] = Field(
        None, description="The directory for this CP2K task"
    )
    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp for this task document was last updated",
    )
    completed_at: Optional[str] = Field(
        None, description="Timestamp for when this task was completed"
    )
    input: Optional[InputSummary] = Field(
        None, description="The input to the first calculation"
    )
    output: Optional[OutputSummary] = Field(
        None, description="The output of the final calculation"
    )
    structure: Union[Structure, Molecule] = Field(
        None, description="Final output structure from the task"
    )
    state: Optional[Status] = Field(None, description="State of this task")
    included_objects: Optional[list[Cp2kObject]] = Field(
        None, description="list of CP2K objects included with this task document"
    )
    cp2k_objects: Optional[dict[Cp2kObject, Any]] = Field(
        None, description="CP2K objects associated with this task"
    )
    entry: Optional[ComputedEntry] = Field(
        None, description="The ComputedEntry from the task doc"
    )
    analysis: Optional[AnalysisSummary] = Field(
        None, description="Summary of structural relaxation and forces"
    )
    run_stats: Optional[dict[str, RunStatistics]] = Field(
        None,
        description="Summary of runtime statistics for each calculation in this task",
    )
    orig_inputs: Optional[dict[str, Cp2kInput]] = Field(
        None, description="Summary of the original CP2K inputs written by custodian"
    )
    task_label: Optional[str] = Field(None, description="A description of the task")
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
        None, description="The inputs and outputs for all CP2K runs in this task."
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
    schema: str = Field(
        __version__, description="Version of atomate2 used to create the document"
    )

    @classmethod
    def from_directory(
        cls: type[_T],
        dir_name: Union[Path, str],
        volumetric_files: tuple[str, ...] = _VOLUMETRIC_FILES,
        store_additional_json: bool = SETTINGS.CP2K_STORE_ADDITIONAL_JSON,
        additional_fields: dict[str, Any] = None,
        **cp2k_calculation_kwargs,
    ) -> "TaskDocument":
        """
        Create a task document from a directory containing CP2K files.

        Parameters
        ----------
        dir_name
            The path to the folder containing the calculation outputs.
        store_additional_json
            Whether to store additional JSON files found in the calculation directory.
        volumetric_files
            Volumetric files to search for.
        additional_fields
            dictionary of additional fields to add to output document.
        **cp2k_calculation_kwargs
            Additional parsing options that will be passed to the
            :obj:`.Calculation.from_cp2k_files` function.

        Returns
        -------
        Cp2kTaskDoc
            A task document for the calculation.
        """
        logger.info(f"Getting task doc in: {dir_name}")

        additional_fields = additional_fields or {}
        dir_name = Path(dir_name)
        task_files = _find_cp2k_files(dir_name, volumetric_files=volumetric_files)

        if len(task_files) == 0:
            raise FileNotFoundError("No CP2K files found!")

        calcs_reversed = []
        all_cp2k_objects = []
        for task_name, files in task_files.items():
            calc_doc, cp2k_objects = Calculation.from_cp2k_files(
                dir_name, task_name, **files, **cp2k_calculation_kwargs
            )
            calcs_reversed.append(calc_doc)
            all_cp2k_objects.append(cp2k_objects)

        analysis = AnalysisSummary.from_cp2k_calc_docs(calcs_reversed)
        transformations, icsd_id, tags, author = parse_transformations(dir_name)
        if tags:
            tags.extend(additional_fields.get("tags", []))
        else:
            tags = additional_fields.get("tags")
        custodian = parse_custodian(dir_name)
        orig_inputs = _parse_orig_inputs(dir_name)

        additional_json = None
        if store_additional_json:
            additional_json = parse_additional_json(dir_name)

        dir_name = get_uri(dir_name)  # convert to full uri path

        # only store objects from last calculation
        # TODO: make this an option
        cp2k_objects = all_cp2k_objects[-1]
        included_objects = None
        if cp2k_objects:
            included_objects = list(cp2k_objects)

        if isinstance(calcs_reversed[0].output.structure, Structure):
            attr = "from_structure"
            dat = {
                "structure": calcs_reversed[0].output.structure,
                "meta_structure": calcs_reversed[0].output.structure,
                "include_structure": True,
            }
        elif isinstance(calcs_reversed[0].output.structure, Molecule):
            attr = "from_molecule"
            dat = {
                "structure": calcs_reversed[0].output.structure,
                "meta_structure": calcs_reversed[0].output.structure,
                "molecule": calcs_reversed[0].output.structure,
                "include_molecule": True,
            }

        doc = getattr(cls, attr)(**dat)
        data = {
            "structure": calcs_reversed[0].output.structure,
            "meta_structure": calcs_reversed[0].output.structure,
            "dir_name": dir_name,
            "calcs_reversed": calcs_reversed,
            "analysis": analysis,
            "transformations": transformations,
            "custodian": custodian,
            "orig_inputs": orig_inputs,
            "additional_json": additional_json,
            "icsd_id": icsd_id,
            "tags": tags,
            "author": author,
            "completed_at": calcs_reversed[0].completed_at,
            "input": InputSummary.from_cp2k_calc_doc(calcs_reversed[0]),
            "output": OutputSummary.from_cp2k_calc_doc(calcs_reversed[0]),
            "state": _get_state(calcs_reversed, analysis),
            "entry": cls.get_entry(calcs_reversed),
            "run_stats": _get_run_stats(calcs_reversed),
            "cp2k_objects": cp2k_objects,
            "included_objects": included_objects,
        }
        doc = cls(**doc.dict())
        doc = doc.copy(update=data)
        return doc.copy(update=additional_fields)

    @staticmethod
    def get_entry(
        calc_docs: list[Calculation], job_id: Optional[str] = None
    ) -> ComputedEntry:
        """
        Get a computed entry from a list of CP2K calculation documents.

        Parameters
        ----------
        calc_docs
            A list of CP2K calculation documents.
        job_id
            The job identifier.

        Returns
        -------
        ComputedEntry
            A computed entry.
        """
        entry_dict = {
            "correction": 0.0,
            "entry_id": job_id,
            "composition": calc_docs[-1].output.structure.composition,
            "energy": calc_docs[-1].output.energy,
            "parameters": {
                # Required to be compatible with MontyEncoder for the ComputedEntry
                "run_type": str(calc_docs[-1].run_type),
            },
            "data": {
                "last_updated": datetime_str(),
            },
        }
        return ComputedEntry.from_dict(entry_dict)


def _find_cp2k_files(
    path: Union[str, Path],
    volumetric_files: tuple[str, ...] = _VOLUMETRIC_FILES,
) -> dict[str, Any]:
    """
    Find CP2K files in a directory.

    Only files in folders with names matching a task name (or alternatively files
    with the task name as an extension, e.g., vasprun.relax1.xml) will be returned.

    CP2K files in the current directory will be given the task name "standard".

    Parameters
    ----------
    path
        Path to a directory to search.
    volumetric_files
        Volumetric files to search for.

    Returns
    -------
    dict[str, Any]
        The filenames of the calculation outputs for each CP2K task, given as a ordered
        dictionary of::

            {
                task_name: {
                    "cp2k_output_file": cp2k_output_filename,
                    "volumetric_files": [v_hartree file, e_density file, etc],
    """
    task_names = ["precondition"] + [f"relax{i}" for i in range(9)]
    path = Path(path)
    task_files = OrderedDict()

    def _get_task_files(files: list[Path], suffix: str = "") -> dict[str, Any]:
        cp2k_files: dict[str, Any] = {}
        vol_files = []
        for file in files:
            if file.match(f"*cp2k.out{suffix}*"):
                cp2k_files["cp2k_output_file"] = Path(file).name
        for vol in volumetric_files:
            _files = [f.name for f in files if f.match(f"*{vol}*cube{suffix}*")]
            _files.sort(key=natural_keys, reverse=True)
            if _files:
                vol_files.append(_files[0])

        if len(vol_files) > 0:
            # add volumetric files if some were found or other cp2k files were found
            cp2k_files["volumetric_files"] = vol_files

        return cp2k_files

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


def _parse_orig_inputs(dir_name: Path) -> dict[str, Cp2kInput]:
    """
    Parse original input files.

    Calculations using custodian generate a *.orig file for the inputs. This is useful
    to know how the calculation originally started.

    Parameters
    ----------
    dir_name
        Path to calculation directory.

    Returns
    -------
    dict[str, Cp2kInput]
        The original data.
    """
    orig_inputs = {}
    input_mapping = {
        "input": {
            "filename": "cp2k.inp",
            "object": Cp2kInput,
        }
    }
    for filename in dir_name.glob("*.orig*"):
        for name, cp2k_input in input_mapping.items():
            fn = cp2k_input.get("filename")
            obj = cp2k_input.get("object")
            if f"{fn}.orig" in str(filename):
                orig_inputs[name.lower()] = obj.from_file(filename)

    return orig_inputs


def _get_max_force(calc_doc: Calculation) -> Optional[float]:
    """Get max force acting on atoms from a calculation document."""
    forces = (
        calc_doc.output.ionic_steps[-1].get("forces")
        if calc_doc.output.ionic_steps
        else None
    )
    structure = calc_doc.output.structure
    if forces:
        forces = np.array(forces)
        sdyn = structure.site_properties.get("selective_dynamics")
        if sdyn:
            forces[np.logical_not(sdyn)] = 0
        return max(np.linalg.norm(forces, axis=1))
    return None


def _get_state(calc_docs: list[Calculation], analysis: AnalysisSummary) -> Status:
    """Get state from calculation documents and relaxation analysis."""
    all_calcs_completed = all(c.has_cp2k_completed == Status.SUCCESS for c in calc_docs)
    if len(analysis.errors) == 0 and all_calcs_completed:
        return Status.SUCCESS  # type: ignore[return-value]
    return Status.FAILED  # type: ignore[return-value]


def _get_run_stats(calc_docs: list[Calculation]) -> dict[str, RunStatistics]:
    """Get summary of runtime statistics for each calculation in this task."""
    run_stats = {}
    total = {
        "total_time": 0.0,
    }
    for calc_doc in calc_docs:
        stats = calc_doc.output.run_stats
        run_stats[calc_doc.task_name] = stats
        total["total_time"] += stats.total_time
    run_stats["overall"] = RunStatistics(**total)
    return run_stats
