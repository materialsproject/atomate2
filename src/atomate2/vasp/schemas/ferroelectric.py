"""Schemas for Ferroelectric wflow."""

import numpy as np
from monty.serialization import dumpfn
from pydantic import BaseModel, Field
from pymatgen.analysis.ferroelectricity.polarization import EnergyTrend, Polarization
from pymatgen.core import Structure
from typing_extensions import Self

__all__ = ["PolarizationDocument"]


class PolarizationDocument(BaseModel):
    """Symmetry data set for materials documents."""

    pretty_formula: str = Field(
        None,
        title="Pretty Formula",
        description="Cleaned representation of the formula",
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

    polarization_change_norm: float = Field(
        None,
        title="Polarization Vector Norm",
        description="The norm of the polarization vector",
    )

    same_branch_polarization: dict = Field(
        None,
        title="Same Branch Polarization Vectors",
        description="Polarization vectors in the same branch",
    )

    raw_electron_polarization: dict = Field(
        None,
        title="Raw Electron Polarization",
        description="Electronic contribution to the polarization",
    )

    raw_ion_polarization: dict = Field(
        None,
        title="Raw Ions Polarization",
        description="Ionic contribution to the polarization",
    )

    polarization_quanta: dict = Field(
        None,
        title="Polarization Quanta",
        description="Quanta of polarization for each structure and direction",
    )

    zval_dict: dict = Field(
        None,
        title="Atomic Z values",
        description="Charge of the atoms as in pseudopotentials",
    )

    energies: list[float] = Field(
        None,
        title="Energies",
        description="Total energy of each structure",
    )

    energies_per_atom: list[float] = Field(
        None, title="Total energy per atom of each structure", description=""
    )

    structures: list[Structure] = Field(
        None,
        title="Structures",
        description="All the interpolated structures",
    )

    polarization_max_spline_jumps: list[float] = Field(
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

    uuids: list[str] = Field(None, description="The uuids of the polarization jobs.")

    job_dirs: list[str] = Field(
        None, description="The directories where the polarization jobs were run."
    )

    @classmethod
    def from_pol_output(
        cls,
        p_elecs: list[float],
        p_ions: list[float],
        structures: list[Structure],
        energies: list[float],
        energies_per_atom: list[float],
        zval_dict: dict,
        tasks: list[str],
        job_dirs: list[str],
        uuids: list[str],
    ) -> Self:
        """
        Generate a PolarizationDocument from output of lcalcpol calculations.

        Parameters
        ----------
        p_elecs: List[float]
            Electronic dipoles
        p_ions: List[float]
            Ionic dipoles
        structures: List[Structure]
            Structures in the order nonpolar, interpolated, polar
        energies: List[float]
            total energy for each calculation
        energies_per_atom: List[float]
            Total energy per atom for each calculation
        zval_dict: Dict
            zvals from pseudopotentials
        tasks: List[str]
            Labels of each polarization task calculation
        job_dirs: List[str]
            The directories where the polarization jobs were run
        uuids: List[str]
            The uuids of the polarization jobs
        """
        polarization = Polarization(p_elecs, p_ions, structures)

        p_change = np.ravel(polarization.get_polarization_change()).tolist()
        p_norm = polarization.get_polarization_change_norm()
        same_branch = polarization.get_same_branch_polarization_data(
            convert_to_muC_per_cm2=True
        )
        raw_elecs, raw_ions = polarization.get_pelecs_and_pions()
        quanta = polarization.get_lattice_quanta(convert_to_muC_per_cm2=True)

        if len(structures) > 3:
            energy_trend = EnergyTrend(energies_per_atom)
            energy_max_spline_jumps = energy_trend.max_spline_jump()
            polarization_max_spline_jumps = polarization.max_spline_jumps()
        else:
            energy_max_spline_jumps = None
            polarization_max_spline_jumps = None

        polarization_dict = {}

        def split_abc(var: np.array) -> dict[str, list[float]]:
            d = {}
            for i, j in enumerate("abc"):
                d.update({f"{j}": np.ravel(var[:, i].tolist())})
            return d

        # General information
        polarization_dict.update(
            {"pretty_formula": structures[0].composition.reduced_formula}
        )
        # polarization_dict.update({"wfid": wfid})
        polarization_dict.update({"task_label_order": tasks})

        # Polarization information
        polarization_dict.update({"polarization_change": p_change})
        polarization_dict.update({"polarization_change_norm": p_norm})
        polarization_dict.update(
            {"polarization_max_spline_jumps": polarization_max_spline_jumps}
        )
        polarization_dict.update({"same_branch_polarization": split_abc(same_branch)})
        polarization_dict.update({"raw_electron_polarization": split_abc(raw_elecs)})
        polarization_dict.update({"raw_ion_polarization": split_abc(raw_ions)})
        polarization_dict.update({"polarization_quanta": split_abc(quanta)})
        polarization_dict.update({"zval_dict": zval_dict})

        # Energy information
        polarization_dict.update(
            {"energy_per_atom_max_spline_jumps": energy_max_spline_jumps}
        )
        polarization_dict.update({"energies": energies})
        polarization_dict.update({"energies_per_atom": energies_per_atom})
        polarization_dict.update({"structures": structures})

        # Add job_dirs and uuids to the polarization_dict
        polarization_dict.update({"job_dirs": job_dirs})
        polarization_dict.update({"uuids": uuids})

        dumpfn(polarization_dict, "polarization_doc.json")
        return cls(**polarization_dict)
