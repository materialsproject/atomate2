"""Schemas for crystal symmetry."""

from typing import Any, Dict

from jobflow.utils import ValueEnum
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, spglib
from pymatgen.core.structure import Structure

from atomate2 import SETTINGS

__all__ = ["PolarizationDocument"]


class PolarizationDocument(BaseModel):
    """Symmetry data set for materials documents."""

    formula_pretty: str = Field(
        None,
        title="Pretty Formula",
        description="Cleaned representation of the formula",
    )

    wfid: str = Field(
        None, title="WF id", description="The workflow id"
    )

    task_label_order: list[str] = Field(
        None,
        title="Task Labels Ordered",
        description="Task labels in order from nonpolar to polar",
    )

    polarization_change: list[float] = Field(
        None,
        title="Polarization Vector",
        description="The polarization vector",
    )

    polarization_norm: float = Field(
        None,
        title="Polarization Vector Norm",
        description="The norm of the polarization vector",
    )

    same_branch_polarization: list[list[float]]  = Field(
        None, title="Same Branch Polarization Vectors",
        description="Polarization vectors in the same branch",
    )

    raw_electron_polarization:  list[float] = Field(
        None, title="Raw Electron Polarization",
        description="Electronic contribution to the polarization",
    )

    raw_ion_polarization:  list[float] = Field(
        None, title="Raw Ions Polarization",
        description="Ionic contribution to the polarization",
    )

    polarization_quanta: list[list[float]] = Field(
        None, title="Polarization Quanta",
        description="Quanta of polarization for each structure and direction",
    )

    zval_dict: dict = Field(
        None, title="Atomic Z values",
        description="Charge of the atoms as in pseudopotentials",
    )

    energies: list[float] = Field(
        None, title="Energies",
        description="Total energy of each structure",
    )

    energies_per_atom: list[float] = Field(
        None, title="Total energy per atom of each structure",
        description=""
    )

    outcars: dict = Field(
        None, title="Outcars",
        description="VASP Outcar for each structure",
    )

    structures: list[Structure] = Field(
        None, title="Structures",
        description="All the interpolated structures",
    )

    polarization_max_spline_jumps: float = Field(
        None, title="Polarization Max Spline Jump",
        description="Maximum jump of the spline that interpolate the polarization branch",
    )

    energy_per_atom_max_spline_jumps: float = Field(
        None, title="Energy Max Spline Jump",
        description="Maximum jump of the spline that interpolate the energy per atom profile",
    )
