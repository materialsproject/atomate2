"""Schemas for Ferroelectric wflow."""

from typing import List

from pydantic import BaseModel, Field
from pymatgen.core import Structure

__all__ = ["PolarizationDocument"]


class PolarizationDocument(BaseModel):
    """Symmetry data set for materials documents."""

    pretty_formula: str = Field(
        None,
        title="Pretty Formula",
        description="Cleaned representation of the formula",
    )

    wfid: str = Field(None, title="WF id", description="The workflow id")

    task_label_order: List[str] = Field(
        None,
        title="Task Labels Ordered",
        description="Task labels in order from nonpolar to polar",
    )

    polarization_change: List[float] = Field(
        None,
        title="Polarization Vector",
        description="The polarization vector",
    )

    polarization_change_norm: float = Field(
        None,
        title="Polarization Vector Norm",
        description="The norm of the polarization vector",
    )

    same_branch_polarization: List[List[float]] = Field(
        None,
        title="Same Branch Polarization Vectors",
        description="Polarization vectors in the same branch",
    )

    raw_electron_polarization: List[float] = Field(
        None,
        title="Raw Electron Polarization",
        description="Electronic contribution to the polarization",
    )

    raw_ion_polarization: List[float] = Field(
        None,
        title="Raw Ions Polarization",
        description="Ionic contribution to the polarization",
    )

    polarization_quanta: List[List[float]] = Field(
        None,
        title="Polarization Quanta",
        description="Quanta of polarization for each structure and direction",
    )

    zval_dict: dict = Field(
        None,
        title="Atomic Z values",
        description="Charge of the atoms as in pseudopotentials",
    )

    energies: List[float] = Field(
        None,
        title="Energies",
        description="Total energy of each structure",
    )

    energies_per_atom: List[float] = Field(
        None, title="Total energy per atom of each structure", description=""
    )

    outcars: List[dict] = Field(
        None,
        title="Outcars",
        description="VASP Outcar for each structure",
    )

    structures: List[Structure] = Field(
        None,
        title="Structures",
        description="All the interpolated structures",
    )

    polarization_max_spline_jumps: List[float] = Field(
        None,
        title="Polarization Max Spline Jump",
        description="Maximum jump of the spline that interpolate \
                     the polarization branch",
    )

    energy_per_atom_max_spline_jumps: float = Field(
        None,
        title="Energy Max Spline Jump",
        description="Maximum jump of the spline that interpolate \
                     the energy per atom profile",
    )
