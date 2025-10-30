"""Schema definitions for ANADDB output documents."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import abipy.core.abinit_units as abu
import numpy as np
from abipy.dfpt.anaddbnc import AnaddbNcFile
from abipy.dfpt.phonons import PhononBands, PhononDos
from abipy.flowtk import events
from abipy.flowtk.utils import File
from emmet.core.math import Matrix3D
from emmet.core.structure import StructureMetadata
from emmet.core.utils import get_num_formula_units
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos as pmgPhononDos

from atomate2.abinit.schemas.calculation import AbinitObject
from atomate2.abinit.schemas.outfiles import AbinitStoredFile
from atomate2.abinit.utils.common import get_event_report
from atomate2.utils.path import get_uri

logger = logging.getLogger(__name__)

__all__ = ["AnaddbTaskDoc", "OutputDoc"]


class OutputDoc(BaseModel):
    """Summary of the outputs for an ANADDB calculation.

    Attributes
    ----------
    structure : Structure or None
        The final pymatgen Structure of the system.
    dijk : list or None
        The conventional static SHG tensor in pm/V (Chi^(2)/2), shape (3, 3, 3).
    e_electronic : list or None
        The electronic contribution to the dielectric tensor, shape (3, 3).
    phonon_bandstructure : PhononBandStructureSymmLine or None
        The phonon band structure object.
    phonon_dos : PhononDos or None
        The phonon density of states object.
    free_energies : list[float] or None
        The vibrational part of the free energies in J/mol per formula unit
        for temperatures in temperature_list.
    heat_capacities : list[float] or None
        The heat capacities in J/K/mol per formula unit for temperatures
        in temperature_list.
    internal_energies : list[float] or None
        The internal energies in J/mol per formula unit for temperatures
        in temperature_list.
    entropies : list[float] or None
        The entropies in J/(K*mol) per formula unit for temperatures
        in temperature_list.
    temperatures : list[int] or None
        The temperatures at which the vibrational part of the free energies
        and other properties have been computed.
    volume_per_formula_unit : float or None
        The volume per formula unit in Angstrom**3.
    formula_units : int or None
        The number of formula units per cell.
    has_imaginary_modes : bool or None
        Whether the structure has imaginary phonon modes.
    born : list[Matrix3D] or None
        The Born effective charges. Only for symmetrically distinct atoms.
    """

    structure: Structure | None = Field(
        None, description="The final structure from the calculation"
    )
    dijk: list | None = Field(
        None, description="Conventional SHG tensor in pm/V (Chi^(2)/2)"
    )
    e_electronic: list | None = Field(
        None, description="Electronic contribution to the dielectric tensor"
    )
    phonon_bandstructure: PhononBandStructureSymmLine | None = Field(
        None,
        description="Phonon band structure object.",
    )
    phonon_dos: pmgPhononDos | None = Field(
        None,
        description="Phonon density of states object.",
    )
    free_energies: list[float] | None = Field(
        None,
        description="vibrational part of the free energies in J/mol per "
        "formula unit for temperatures in temperature_list",
    )
    heat_capacities: list[float] | None = Field(
        None,
        description="heat capacities in J/K/mol per "
        "formula unit for temperatures in temperature_list",
    )
    internal_energies: list[float] | None = Field(
        None,
        description="internal energies in J/mol per "
        "formula unit for temperatures in temperature_list",
    )
    entropies: list[float] | None = Field(
        None,
        description="entropies in J/(K*mol) per formula unit"
        "for temperatures in temperature_list ",
    )
    temperatures: list[int] | None = Field(
        None,
        description="temperatures at which the vibrational"
        " part of the free energies"
        " and other properties have been computed",
    )
    volume_per_formula_unit: float | None = Field(
        None, description="volume per formula unit in Angstrom**3"
    )

    formula_units: int | None = Field(None, description="Formula units per cell")

    has_imaginary_modes: bool | None = Field(
        None, description="if true, structure has imaginary modes"
    )

    born: list[Matrix3D] | None = Field(
        None,
        description="Born charges as computed from phonopy. Only for symmetrically "
        "different atoms",
    )

    @classmethod
    def from_abinit_files(
        cls,
        dir_name: Path | str,
        abinit_anaddb_file: Path | str = "out_anaddb.nc",
        abinit_analog_file: Path | str = "run.log",  # noqa: ARG003
        abinit_phbst_file: Path | str = "out_PHBST.nc",
        abinit_phdos_file: Path | str = "out_PHDOS.nc",
    ) -> OutputDoc:
        """
        Create an ANADDB calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name : Path or str
            The directory containing the calculation outputs.
        abinit_anaddb_file : Path or str
            Path to the ANADDB netCDF output file, relative to dir_name.
            Default is "out_anaddb.nc".
        abinit_analog_file : Path or str
            Path to the main log file of the ANADDB job, relative to dir_name.
            Default is "run.log".
        abinit_phbst_file : Path or str
            Path to the phonon band structure file, relative to dir_name.
            Default is "out_PHBST.nc".
        abinit_phdos_file : Path or str
            Path to the phonon DOS file, relative to dir_name.
            Default is "out_PHDOS.nc".

        Returns
        -------
        OutputDoc
            An ANADDB output document.
        """
        dir_name = Path(dir_name)
        abinit_anaddb_file = dir_name / abinit_anaddb_file
        abinit_phbst_file = dir_name / abinit_phbst_file
        abinit_phdos_file = dir_name / abinit_phdos_file

        if abinit_anaddb_file.exists():
            abinit_anaddb = AnaddbNcFile.from_file(abinit_anaddb_file)
        else:
            raise RuntimeError(
                f"The file {abinit_anaddb_file} is missing and is required \
                to generate the output document"
            )
        if abinit_phbst_file.exists():
            abinit_phbst = PhononBands.from_file(abinit_phbst_file)
        else:
            abinit_phbst = None
        if abinit_phdos_file.exists():
            abinit_phdos = PhononDos.as_phdos(str(abinit_phdos_file))
        else:
            abinit_phdos = None

        structure = abinit_anaddb.structure

        if abinit_phbst:
            phonon_bandstructure = abinit_phbst.to_pymatgen()
            phonon_bandstructure.labels_dict = {
                k.strip("$"): v for k, v in phonon_bandstructure.labels_dict.items()
            }
        else:
            phonon_bandstructure = None
        phonon_dos = abinit_phdos.to_pymatgen() if abinit_phdos else None

        if phonon_dos:
            temperatures = [int(t) for t in abinit_phdos.get_free_energy().mesh]
            free_energies = [
                phonon_dos.helmholtz_free_energy(temp, structure=structure)
                for temp in temperatures
            ]
            heat_capacities = [
                phonon_dos.cv(temp=temp, structure=structure) for temp in temperatures
            ]
            internal_energies = [
                phonon_dos.internal_energy(temp, structure=structure)
                for temp in temperatures
            ]
            entropies = [
                phonon_dos.entropy(temp, structure=structure) for temp in temperatures
            ]
        else:
            temperatures = None
            free_energies = None
            heat_capacities = None
            internal_energies = None
            entropies = None
        born = getattr(abinit_anaddb, "bec", None)
        born = born.values if born else None

        formula_units = get_num_formula_units(structure.composition)

        volume_per_formula_unit = structure.volume / formula_units

        has_imaginary_modes = (
            phonon_bandstructure.has_imaginary_freq() if phonon_bandstructure else None
        )

        # for pm/V units (SI)
        dijk = (
            list(
                abinit_anaddb.dchide
                * 16
                * np.pi**2
                * abu.Bohr_Ang**2
                * 1e-8
                * abu.eps0
                / abu.e_Cb
            )
            if abinit_anaddb.dchide is not None and abinit_anaddb.dchide.any()
            else None
        )
        e_electronic = (
            list(abinit_anaddb.epsinf)
            if abinit_anaddb.epsinf is not None and abinit_anaddb.epsinf.any()
            else None
        )
        return cls(
            structure=structure,
            phonon_bandstructure=phonon_bandstructure,
            phonon_dos=phonon_dos,
            free_energies=free_energies,
            heat_capacities=heat_capacities,
            internal_energies=internal_energies,
            entropies=entropies,
            temperatures=temperatures,
            volume_per_formula_unit=volume_per_formula_unit,
            formula_units=formula_units,
            has_imaginary_modes=has_imaginary_modes,
            born=born,
            dijk=dijk,
            e_electronic=e_electronic,
        )


class AnaddbTaskDoc(StructureMetadata, extra="allow"):  # type: ignore[call-arg]
    """Task document for an ANADDB job.

    Attributes
    ----------
    dir_name : str or None
        The directory for this ANADDB task.
    completed_at : str or None
        Timestamp for when this task was completed.
    output : OutputDoc or None
        The output document of the ANADDB calculation.
    structure : Structure or None
        Final output structure from the task.
    event_report : EventReport or None
        Event report from the ANADDB job.
    included_objects : list[AbinitObject] or None
        List of ABINIT objects included with this task document.
    abinit_objects : dict[AbinitObject, Any] or None
        ABINIT objects associated with this task.
    task_label : str or None
        A description of the task.
    tags : list[str] or None
        Metadata tags for this task document.
    """

    dir_name: str | None = Field(None, description="The directory for this Abinit task")
    completed_at: str | None = Field(
        None, description="Timestamp for when this task was completed"
    )
    output: OutputDoc | None = Field(
        None, description="The output of the final calculation"
    )
    structure: Structure | None = Field(
        None, description="Final output atoms from the task"
    )
    event_report: events.EventReport | None = Field(
        None, description="Event report of this abinit job."
    )
    included_objects: list[AbinitObject] | None = Field(
        None, description="List of Abinit objects included with this task document"
    )
    abinit_objects: dict[AbinitObject, Any] | None = Field(
        None, description="Abinit objects associated with this task"
    )
    task_label: str | None = Field(None, description="A description of the task")
    tags: list[str] | None = Field(
        None, description="Metadata tags for this task document"
    )

    @classmethod
    def from_directory(
        cls,
        dir_name: Path | str,
        additional_fields: dict[str, Any] | None = None,
        abinit_phbst_file: Path | str = "out_PHBST.nc",
        abinit_phdos_file: Path | str = "out_PHDOS.nc",
        abinit_analog_file: Path | str = "run.log",
        files_to_store: list[str] | None = None,
        **output_doc_kwargs,
    ) -> AnaddbTaskDoc:
        """Create a task document from a directory containing ANADDB files.

        Parameters
        ----------
        dir_name : Path or str
            The path to the folder containing the calculation outputs.
        additional_fields : dict[str, Any] or None
            Dictionary of additional fields to add to the output document.
            Default is None.
        abinit_phbst_file : Path or str
            Path to the phonon band structure file, relative to dir_name.
            Default is "out_PHBST.nc".
        abinit_phdos_file : Path or str
            Path to the phonon DOS file, relative to dir_name.
            Default is "out_PHDOS.nc".
        abinit_analog_file : Path or str
            Path to the main log file of the ANADDB job, relative to dir_name.
            Default is "run.log".
        files_to_store : list[str] or None
            List of file types to store (e.g., ["PHBST", "PHDOS"]).
            Default is None.
        **output_doc_kwargs
            Additional keyword arguments passed to OutputDoc.from_abinit_files().

        Returns
        -------
        AnaddbTaskDoc
            A task document for the ANADDB calculation.
        """
        logger.info(f"Getting task doc in: {dir_name}")

        if additional_fields is None:
            additional_fields = {}

        dir_name = Path(dir_name)
        task_files = _find_abinit_files(dir_name)

        if len(task_files) == 0:
            raise FileNotFoundError("No anaddb files found!")
        if len(task_files) > 1:
            raise RuntimeError(
                f"Only one anaddb calculation expected. Found {len(task_files)}"
            )

        std_task_files = next(iter(task_files.values()))

        report = get_event_report(ofile=File(abinit_analog_file))
        if not report.run_completed:
            raise RuntimeError("Anaddb did not complete successfully")

        output_doc = OutputDoc.from_abinit_files(
            dir_name, **std_task_files, **output_doc_kwargs
        )

        phbst_filepath = dir_name / abinit_phbst_file
        abinit_objects: dict[AbinitObject, Any] = {}
        if phbst_filepath.exists() and "PHBST" in files_to_store:
            abinit_objects[AbinitObject.PHBSTFILE] = AbinitStoredFile.from_file(  # type: ignore[index]
                filepath=phbst_filepath, data_type=bytes
            )
        phdos_filepath = dir_name / abinit_phdos_file
        if phdos_filepath.exists() and "PHDOS" in files_to_store:
            abinit_objects[AbinitObject.PHDOSFILE] = AbinitStoredFile.from_file(  # type: ignore[index]
                filepath=phdos_filepath, data_type=bytes
            )
        completed_at = str(
            datetime.fromtimestamp(
                os.stat(abinit_analog_file).st_mtime, tz=timezone.utc
            )
        )

        tags = additional_fields.pop("tags", None)

        dir_name = get_uri(dir_name)  # convert to full uri path

        included_objects = None
        if abinit_objects:
            included_objects = list(abinit_objects.keys())

        data = {
            "dir_name": dir_name,
            "completed_at": completed_at,
            "output": output_doc,
            "event_report": report,
            "included_objects": included_objects,
            "abinit_objects": abinit_objects,
            "tags": tags,
        }

        return cls.from_structure(
            structure=output_doc.structure,
            meta_structure=output_doc.structure,
            **data,
            **additional_fields,
        )


def _find_abinit_files(
    path: Path | str,
) -> dict[str, Any]:
    """
    Find ANADDB output files in a directory.

    This function searches for ANADDB output files (anaddb.nc, PHBST.nc,
    PHDOS.nc, and run.log) in the specified directory and its outdata
    subdirectory.

    Parameters
    ----------
    path : Path or str
        The directory to search for ANADDB output files.

    Returns
    -------
    dict[str, Any]
        A dictionary with task files organized by calculation type.
        Keys are task identifiers (e.g., "standard"), values are dicts
        mapping file types to their relative paths.
    """
    path = Path(path)
    task_files = {}

    def _get_task_files(files: list[Path], suffix: str = "") -> dict:
        abinit_files = {}
        for file in files:
            # Here we make assumptions about the output file naming
            if file.match(f"*outdata/out_anaddb.nc{suffix}*"):
                abinit_files["abinit_anaddb_file"] = Path(file).relative_to(path)
            elif file.match(f"*run.log{suffix}*"):
                abinit_files["abinit_analog_file"] = Path(file).relative_to(path)
            if file.match(f"*outdata/out_PHBST.nc{suffix}*"):
                abinit_files["abinit_phbst_file"] = Path(file).relative_to(path)
            if file.match(f"*outdata/out_PHDOS.nc{suffix}*"):
                abinit_files["abinit_phdos_file"] = Path(file).relative_to(path)

        return abinit_files

    # get any matching file from the root folder
    standard_files = _get_task_files(
        list(path.glob("*")) + list(path.glob("outdata/*"))
    )
    if len(standard_files) > 0:
        task_files["standard"] = standard_files

    return task_files
