"""Schemas for Ferroelectric wflow."""

from typing import Dict, List

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

    @classmethod
    def from_pol_output(
            p_elecs: List[float],
            p_ion: List[float],
            structures: List[Structure],
            energies: List[float],
            energies_per_atom: List[float],
            zval_dicts: List[float],
            tasks: List[str],
    ):

        """
        Generate a PolarizationDocument from output of lcalcpol calculations

        Parameters
        ----------
        p_elecs : List[float]
            electronic dipoles
        p_ion : List[float]
            ionic dipoles
        structures: List[Structure]
            Structures in the order nonpolar, interpolated, polar
        energies: List[float]
            total energy for each calculation
        energies_per_atom: List[float]
            total energy per atom for each calculation        
        zval_dicts: Dict
            zvals from pseudopotentials
        tasks: List[str],
            labels of each polarization task calculation
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

        def split_abc(var):
            d = {}
            for i, j in enumerate("abc"):
                d.update({f"{j}": np.ravel(var[:, i]).tolist()})
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

        dumpfn(polarization_dict, "polarization_doc.json")
        return cls(**polarization_dict)


