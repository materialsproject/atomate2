"""Core definition of a VASP task document."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from emmet.core.math import Matrix3D, Vector3D
from emmet.core.structure import StructureMetadata
from jobflow import Schema
from pydantic import Field
from pymatgen.core.structure import Structure
from pymatgen.entries.computed_entries import ComputedEntry, __version__
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Potcar

from atomate2.settings import settings
from atomate2.vasp.schemas.calculation import (
    Calculation,
    PotcarSpec,
    RunStatistics,
    Status,
    VaspObject,
)


class AnalysisSummary(Schema):
    """Calculation relaxation summary."""

    delta_volume: float = Field(None, description="Absolute change in volume")
    delta_volume_as_percent: float = Field(
        None, description="Percentage change in volume"
    )
    max_force: float = Field(None, description="Maximum force on the atoms")
    warnings: List[str] = Field(None, description="Warnings from the VASP drone")
    errors: List[str] = Field(None, description="Errors from the VASP drone")

    @classmethod
    def from_vasp_calc_docs(cls, calc_docs: List[Calculation]) -> "AnalysisSummary":
        """
        Create analysis summary from VASP calculation documents.

        Parameters
        ----------
        calc_docs
            VASP calculation documents.

        Returns
        -------
        AnalysisSummary
            The relaxation analysis.
        """
        from atomate2.vasp.schemas.calculation import Status

        initial_vol = calc_docs[0].input.structure.lattice.volume
        final_vol = calc_docs[-1].output.structure.lattice.volume
        delta_vol = final_vol - initial_vol
        percent_delta_vol = 100 * delta_vol / initial_vol
        warnings = []
        errors = []

        if abs(percent_delta_vol) > settings.VASP_VOLUME_CHANGE_WARNING_TOL:
            warnings.append(
                f"Volume change > {settings.VASP_VOLUME_CHANGE_WARNING_TOL * 100}%"
            )

        final_calc = calc_docs[-1]
        max_force = None
        if final_calc.has_vasp_completed == Status.SUCCESS:
            # max force and valid structure checks
            structure = final_calc.output.structure
            max_force = _get_max_force(final_calc)
            warnings.extend(_get_drift_warnings(final_calc))
            if not structure.is_valid():
                errors.append("Bad structure (atoms are too close!)")

        return cls(
            delta_volume=delta_vol,
            delta_volume_as_percent=percent_delta_vol,
            max_force=max_force,
            warnings=warnings,
            errors=errors,
        )


class PseudoPotentialSummary(Schema):
    """A summary of pseudo-potential type and functional."""

    pot_type: str = Field(None, description="Pseudo-potential type, e.g. PAW")
    functional: str = Field(
        None, description="Functional for the pseudo-potential (e.g. PBE)"
    )
    labels: List[str] = Field(
        None, description="Labels of the POTCARs as distributed in VASP"
    )


class InputSummary(Schema):
    """Summary of inputs for a VASP calculation."""

    structure: Structure = Field(None, description="The input structure object")
    parameters: Dict = Field(
        None,
        description="Parameters from vasprun for the last calculation in the series",
    )
    pseudo_potentials: PseudoPotentialSummary = Field(
        None, description="Summary of the pseudo-potentials used in this calculation"
    )
    potcar_spec: List[PotcarSpec] = Field(
        None, description="Title and hash of POTCAR files used in the calculation"
    )
    xc_override: str = Field(
        None, description="Exchange-correlation functional used if not the default"
    )

    @classmethod
    def from_vasp_calc_doc(cls, calc_doc: Calculation) -> "InputSummary":
        """
        Create calculation input summary from a calculation document.

        Parameters
        ----------
        calc_doc
            A VASP calculation document.

        Returns
        -------
        InputSummary
            A summary of the input structure and parameters.
        """
        xc = calc_doc.input.incar.get("GGA")
        if xc:
            xc = xc.upper()

        pot_type, func = calc_doc.input.potcar_type[0].split("_")
        func = "lda" if len(pot_type) == 1 else "_".join(func)
        pps = PseudoPotentialSummary(
            pot_type=pot_type, functional=func, labels=calc_doc.input.potcar
        )

        return cls(
            structure=calc_doc.input.structure,
            parameters=calc_doc.input.parameters,
            pseudo_potentials=pps,
            potcar_spec=calc_doc.input.potcar_spec,
            xc_override=xc,
        )


class OutputSummary(Schema):
    """Summary of the outputs for a VASP calculation."""

    structure: Structure = Field(None, description="The output structure object")
    energy: float = Field(
        None, description="The final total DFT energy for the last calculation"
    )
    energy_per_atom: float = Field(
        None, description="The final DFT energy per atom for the last calculation"
    )
    bandgap: float = Field(None, description="The DFT bandgap for the last calculation")
    forces: List[Vector3D] = Field(
        None, description="Forces on atoms from the last calculation"
    )
    stress: Matrix3D = Field(
        None, description="Stress on the unit cell from the last calculation"
    )

    @classmethod
    def from_vasp_calc_doc(cls, calc_doc: Calculation) -> "OutputSummary":
        """
        Create a summary of VASP calculation outputs from a VASP calculation document.

        Parameters
        ----------
        calc_doc
            A VASP calculation document.

        Returns
        -------
        OutputSummary
            The calculation output summary.
        """
        return cls(
            structure=calc_doc.output.structure,
            energy=calc_doc.output.energy,
            energy_per_atom=calc_doc.output.energy_per_atom,
            bandgap=calc_doc.output.bandgap,
            forces=calc_doc.output.ionic_steps[-1]["forces"],
            stress=calc_doc.output.ionic_steps[-1]["stress"],
        )


class TaskDocument(Schema, StructureMetadata):
    """Definition of VASP task document."""

    dir_name: str = Field(None, description="The directory for this VASP task")
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp for this task document was last updated",
    )
    completed_at: datetime = Field(
        None, description="Timestamp for when this task was completed"
    )
    input: InputSummary = Field(None, description="The input to the first calculation")
    output: OutputSummary = Field(
        None, description="The output of the final calculation"
    )
    structure: Structure = Field(
        None, description="Final output structure from the task"
    )
    state: Status = Field(None, description="State of this task")
    included_objects: List[VaspObject] = Field(
        None, description="List of VASP objects included with this task document"
    )
    vasp_objects: Dict[VaspObject, Any] = Field(
        None, description="Vasp objects associated with this task"
    )
    entry: ComputedEntry = Field(
        None, description="The ComputedEntry from the task doc"
    )
    analysis: AnalysisSummary = Field(
        None, description="Summary of structural relaxation and forces"
    )
    run_stats: Dict[str, RunStatistics] = Field(
        None,
        description="Summary of runtime statistics for each calculation in this task",
    )
    orig_inputs: Dict[str, Union[Kpoints, Incar, Poscar, Potcar]] = Field(
        None, description="Summary of the original VASP inputs writen by custodian"
    )
    task_id: str = Field(None, description="The task identifier for this document")
    task_label: str = Field(None, description="A description of the task")
    tags: List[str] = Field(None, description="Metadata tags for this task document")
    author: str = Field(None, description="Author extracted from transformations")
    icsd_id: str = Field(
        None, description="International crystal structure database id of the structure"
    )
    calcs_reversed: List[Calculation] = Field(
        None, description="The inputs and outputs for all VASP runs in this task."
    )
    transformations: Dict[str, Any] = Field(
        None,
        description="Information on the structural transformations, parsed from a "
        "transformations.json file",
    )
    custodian: Any = Field(
        None,
        description="Information on the custodian settings used to run this "
        "calculation, parsed from a custodian.json file",
    )
    additional_json: Dict[str, Any] = Field(
        None, description="Additional json loaded from the calculation directory"
    )
    _schema: str = Field(
        __version__,
        description="Version of the atomate2 used to create the document",
        alias="schema",
    )

    @classmethod
    def from_task_files(
        cls,
        dir_name: Union[Path, str],
        task_files: Dict[str, Dict[str, Any]],
        store_additional_json: bool = settings.VASP_STORE_ADDITIONAL_JSON,
        **vasp_calc_doc_kwargs,
    ) -> "TaskDocument":
        """
        Create a task document from VASP files.

        Parameters
        ----------
        dir_name
            The path to the folder containing the calculation outputs.
        task_files
            The filenames of the calculation outputs for each VASP task given as an
            ordered dictionary of::

                {
                    task_name: {
                        "vasprun_file": vasprun_filename,
                        "outcar_file": outcar_filename,
                        "volumetric_files": [CHGCAR, LOCPOT, etc]
                    },
                    ...
                }

        store_additional_json
            Whether to store additional json files found in the calculation directory.
        **vasp_calc_doc_kwargs
            Additional parsing options that will be passed to the
            :obj:`.Calculation.from_vasp_files` function.

        Returns
        -------
        VaspTaskDoc
            A task document for the calculation.
        """
        from pathlib import Path

        from atomate2.utils.path import get_uri
        from atomate2.vasp.schemas.calculation import Calculation

        dir_name = Path(dir_name)

        calcs_reversed = []
        all_vasp_objects = []
        for task_name, files in task_files.items():
            calc_doc, vasp_objects = Calculation.from_vasp_files(
                dir_name, task_name, **files, **vasp_calc_doc_kwargs
            )
            calcs_reversed.append(calc_doc)
            all_vasp_objects.append(vasp_objects)

        analysis = AnalysisSummary.from_vasp_calc_docs(calcs_reversed)
        transformations, icsd_id, tags, author = _parse_transformations(dir_name)
        custodian = _parse_custodian(dir_name)
        orig_inputs = _parse_orig_inputs(dir_name)

        additional_json = None
        if store_additional_json:
            additional_json = _parse_additional_json(dir_name)

        dir_name = get_uri(dir_name)  # convert to full uri path

        # only store objects from last calculation
        # TODO: make this an option
        vasp_objects = all_vasp_objects[-1]
        included_objects = None
        if vasp_objects:
            included_objects = list(vasp_objects.keys())

        return cls.from_structure(
            structure=calcs_reversed[-1].output.structure,
            include_structure=True,
            dir_name=dir_name,
            calcs_reversed=calcs_reversed,
            analysis=analysis,
            transformations=transformations,
            custodian=custodian,
            orig_inputs=orig_inputs,
            additional_json=additional_json,
            icsd_id=icsd_id,
            tags=tags,
            author=author,
            completed_at=calcs_reversed[-1].completed_at,
            input=InputSummary.from_vasp_calc_doc(calcs_reversed[0]),
            output=OutputSummary.from_vasp_calc_doc(calcs_reversed[-1]),
            state=_get_state(calcs_reversed, analysis),
            entry=cls.get_entry(calcs_reversed),
            run_stats=_get_run_stats(calcs_reversed),
            vasp_objects=vasp_objects,
            included_objects=included_objects,
        )

    @staticmethod
    def get_entry(
        calc_docs: List[Calculation], task_id: Optional[str] = None
    ) -> ComputedEntry:
        """
        Get a computed entry from a list of VASP calculation documents.

        Parameters
        ----------
        calc_docs
            A list of VASP calculation documents.
        task_id
            The task identifier.

        Returns
        -------
        ComputedEntry
            A computed entry.
        """
        from datetime import datetime

        from pymatgen.analysis.structure_analyzer import oxide_type
        from pymatgen.entries.computed_entries import ComputedEntry

        entry_dict = {
            "correction": 0.0,
            "entry_id": task_id,
            "composition": calc_docs[-1].output.structure.composition,
            "energy": calc_docs[-1].output.energy,
            "parameters": {
                "potcar_spec": calc_docs[-1].input.potcar_spec,
                # Required to be compatible with MontyEncoder for the ComputedEntry
                "run_type": str(calc_docs[-1].run_type),
            },
            "data": {
                "oxide_type": oxide_type(calc_docs[-1].output.structure),
                "last_updated": datetime.utcnow,
            },
        }
        return ComputedEntry.from_dict(entry_dict)


def _parse_transformations(
    dir_name: Path,
) -> Tuple[Dict, Optional[int], Optional[List[str]], Optional[str]]:
    """Parse transformations.json file."""
    from monty.serialization import loadfn

    transformations = {}
    filenames = tuple(dir_name.glob("transformations.json*"))
    icsd_id = None
    if len(filenames) >= 1:
        transformations = loadfn(filenames[0], cls=None)
        try:
            match = re.match(r"(\d+)-ICSD", transformations["history"][0]["source"])
            if match:
                icsd_id = int(match.group(1))
        except (KeyError, IndexError):
            pass

    # We don't want to leave tags or authors in the
    # transformations file because they'd be copied into
    # every structure generated after this one.
    other_parameters = transformations.get("other_parameters", {})
    new_tags = other_parameters.pop("tags", None)
    new_author = other_parameters.pop("author", None)

    if "other_parameters" in transformations and not other_parameters:
        # if dict is now empty remove it
        transformations.pop("other_parameters")

    return transformations, icsd_id, new_tags, new_author


def _parse_custodian(dir_name: Path) -> Optional[Dict]:
    """
    Parse custodian.json file.

    Calculations done using custodian have a custodian.json file which tracks the jobs
    performed and any errors detected and fixed.

    Parameters
    ----------
    dir_name
        Path to calculation directory.

    Returns
    -------
    Optional[dict]
        The information parsed from custodian.json file.
    """
    from monty.serialization import loadfn

    filenames = tuple(dir_name.glob("custodian.json*"))
    if len(filenames) >= 1:
        return loadfn(filenames[0], cls=None)
    return None


def _parse_orig_inputs(
    dir_name: Path,
) -> Dict[str, Union[Kpoints, Poscar, Potcar, Incar]]:
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
    Dict[str, Union[Kpints, Poscar, Potcar, Incar]]
        The original POSCAR, KPOINTS, POTCAR, and INCAR data.
    """
    from pymatgen.io.vasp import Incar, Kpoints, Poscar, Potcar

    orig_inputs = {}
    input_mapping = {
        "INCAR": Incar,
        "KPOINTS": Kpoints,
        "POTCAR": Potcar,
        "POSCAR": Poscar,
    }
    for filename in dir_name.glob("*.orig*"):
        for name, vasp_input in input_mapping.items():
            if f"{name}.orig" in str(filename):
                orig_inputs[name.lower()] = vasp_input.from_file(filename)
    return orig_inputs


def _parse_additional_json(dir_name: Path) -> Dict[str, Any]:
    """Parse additional json files in the directory."""
    from monty.serialization import loadfn

    additional_json = {}
    for filename in dir_name.glob("*.json*"):
        key = filename.name.split(".")[0]
        if key not in ("custodian", "transformations"):
            additional_json[key] = loadfn(filename, cls=None)
    return additional_json


def _get_max_force(calc_doc: Calculation) -> Optional[float]:
    """Get max force acting on atoms from a calculation document."""
    import numpy as np

    forces = calc_doc.output.ionic_steps[-1].get("forces")
    structure = calc_doc.output.structure
    if forces:
        forces = np.array(forces)
        sdyn = structure.site_properties.get("selective_dynamics")
        if sdyn:
            forces[np.logical_not(sdyn)] = 0
        return max(np.linalg.norm(forces, axis=1))
    return None


def _get_drift_warnings(calc_doc: Calculation) -> List[str]:
    """Get warnings of whether the drift on atoms is too large."""
    import numpy as np

    warnings = []
    if calc_doc.input.parameters.get("NSW", 0) > 0:
        drift = calc_doc.output.outcar.get("drift", [[0, 0, 0]])
        max_drift = max([np.linalg.norm(d) for d in drift])
        ediffg = calc_doc.input.parameters.get("EDIFFG", None)
        if ediffg and float(ediffg) < 0:
            max_force = -float(ediffg)
        else:
            max_force = np.inf
        if max_drift > max_force:
            warnings.append(
                f"Drift ({drift}) > desired force convergence ({max_force}), structure "
                "likely not converged to desired accuracy."
            )
    return warnings


def _get_state(calc_docs: List[Calculation], analysis: AnalysisSummary) -> Status:
    """Get state from calculation documents and relaxation analysis."""
    from atomate2.vasp.schemas.calculation import Status

    all_calcs_completed = all(
        [c.has_vasp_completed == Status.SUCCESS for c in calc_docs]
    )
    if len(analysis.errors) == 0 and all_calcs_completed:
        return Status.SUCCESS  # type: ignore
    return Status.FAILED  # type: ignore


def _get_run_stats(calc_docs: List[Calculation]) -> Dict[str, RunStatistics]:
    """Get summary of runtime statistics for each calculation in this task."""
    from atomate2.vasp.schemas.calculation import RunStatistics

    run_stats = {}
    total = dict(
        average_memory=0.0,
        max_memory=0.0,
        elapsed_time=0.0,
        system_time=0.0,
        user_time=0.0,
        total_time=0.0,
        cores=0,
    )
    for calc_doc in calc_docs:
        stats = calc_doc.output.run_stats
        run_stats[calc_doc.task_name] = stats
        total["average_memory"] = max(total["average_memory"], stats.average_memory)
        total["max_memory"] = max(total["max_memory"], stats.max_memory)
        total["cores"] = max(total["cores"], stats.cores)
        total["elapsed_time"] += stats.elapsed_time
        total["system_time"] += stats.system_time
        total["user_time"] += stats.user_time
        total["total_time"] += stats.total_time
    run_stats["overall"] = RunStatistics(**total)
    return run_stats
