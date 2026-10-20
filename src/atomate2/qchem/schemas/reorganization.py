"""Schema for inner-sphere (intramolecular) reorganization energies."""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core import Molecule
from pymatgen.core.units import Ha_to_eV

logger = logging.getLogger(__name__)


class InnerReorganizationDocument(BaseModel):
    """
    Inner-sphere reorganization energy of an electron transfer A^q1 -> A^q2.

    Following the four-point scheme of Nelsen et al., each of the two states is
    evaluated at its own optimized geometry and at the optimized geometry of the
    other state, at the same level of theory. With an equilibrium (or no) solvent
    model this gives the intramolecular (inner-sphere) reorganization energies only;
    the outer-sphere (solvent) contribution depends on the environment (donor-acceptor
    distance, electrode, dielectric constants) and is left to post-processing.

    All energies of the two geometries are single points of the same input set, so
    they are directly comparable. Energies are in eV, except the raw single-point
    energies, which are in Hartree.
    """

    charge_1: int = Field(description="Charge of state 1.")
    spin_multiplicity_1: int = Field(description="Spin multiplicity of state 1.")
    charge_2: int = Field(description="Charge of state 2.")
    spin_multiplicity_2: int = Field(description="Spin multiplicity of state 2.")

    molecule_1: Molecule = Field(description="Optimized geometry of state 1.")
    molecule_2: Molecule = Field(description="Optimized geometry of state 2.")

    energy_1_at_1: float = Field(description="Energy of state 1 at geometry 1 (Ha).")
    energy_1_at_2: float = Field(description="Energy of state 1 at geometry 2 (Ha).")
    energy_2_at_2: float = Field(description="Energy of state 2 at geometry 2 (Ha).")
    energy_2_at_1: float = Field(description="Energy of state 2 at geometry 1 (Ha).")

    lambda_1: float = Field(
        description="Reorganization energy on the potential energy surface of state "
        "1, E_1(R_2) - E_1(R_1) (eV)."
    )
    lambda_2: float = Field(
        description="Reorganization energy on the potential energy surface of state "
        "2, E_2(R_1) - E_2(R_2) (eV)."
    )
    lambda_mean: float = Field(
        description="Mean of lambda_1 and lambda_2, the inner-sphere reorganization "
        "energy assuming parabolas of equal curvature (eV)."
    )

    vertical_energy_1_to_2: float = Field(
        description="Vertical energy change from state 1 to state 2 at geometry 1, "
        "E_2(R_1) - E_1(R_1) (eV): the vertical ionization energy or (minus) electron "
        "affinity."
    )
    vertical_energy_2_to_1: float = Field(
        description="Vertical energy change from state 2 to state 1 at geometry 2, "
        "E_1(R_2) - E_2(R_2) (eV)."
    )
    adiabatic_energy: float = Field(
        description="Adiabatic energy change E_2(R_2) - E_1(R_1) (eV)."
    )
    adiabatic_free_energy: float | None = Field(
        None,
        description="Adiabatic free energy change G_2 - G_1 at 298.15 K (eV), from the "
        "frequency calculations of the optimizations, if available.",
    )

    geometry_rmsd: float = Field(
        description="RMSD between the two optimized geometries after optimal "
        "superposition (Angstrom)."
    )
    connectivity_change: str | None = Field(
        None,
        description="Result of pymatgen's check_for_structure_changes between the two "
        "optimized geometries ('no_change' if the bonding is the same). Anything else "
        "means the electron transfer is coupled to bond breaking or formation, and "
        "the four-point reorganization energy is not meaningful. None if openbabel is "
        "not available.",
    )

    solvation_lot_info: str | None = Field(
        None, description="Level of theory and solvation of the single points."
    )
    single_point_uuids: dict[str, str] | None = Field(
        None, description="UUIDs of the four single-point jobs."
    )
    single_point_dirs: dict[str, str] | None = Field(
        None, description="Directories of the four single-point calculations."
    )

    @classmethod
    def from_energies(
        cls,
        molecule_1: Molecule,
        molecule_2: Molecule,
        energies: dict[str, float],
        free_energy_1: float | None = None,
        free_energy_2: float | None = None,
        **kwargs,
    ) -> InnerReorganizationDocument:
        """
        Create the document from the four single-point energies.

        Parameters
        ----------
        molecule_1, molecule_2 : .Molecule
            Optimized geometries of states 1 and 2, with their charges and spins.
        energies : dict
            Single-point energies (Ha) with keys "1@1", "1@2", "2@2" and "2@1", where
            "i@j" is state i at geometry j.
        free_energy_1, free_energy_2 : float or None
            Free energies of the two optimized states (eV), if available.
        **kwargs
            Other fields of the document.

        Returns
        -------
        InnerReorganizationDocument
            The document.
        """
        e11, e12 = energies["1@1"], energies["1@2"]
        e22, e21 = energies["2@2"], energies["2@1"]
        lambda_1 = (e12 - e11) * Ha_to_eV
        lambda_2 = (e21 - e22) * Ha_to_eV
        if min(lambda_1, lambda_2) < 0:
            logger.warning(
                "Negative reorganization energy (lambda_1 = %.3f eV, "
                "lambda_2 = %.3f eV): "
                "the geometries are probably not both minima of their states.",
                lambda_1,
                lambda_2,
            )
        adiabatic_free_energy = (
            free_energy_2 - free_energy_1
            if free_energy_1 is not None and free_energy_2 is not None
            else None
        )
        return cls(
            charge_1=int(molecule_1.charge),
            spin_multiplicity_1=int(molecule_1.spin_multiplicity),
            charge_2=int(molecule_2.charge),
            spin_multiplicity_2=int(molecule_2.spin_multiplicity),
            molecule_1=molecule_1,
            molecule_2=molecule_2,
            energy_1_at_1=e11,
            energy_1_at_2=e12,
            energy_2_at_2=e22,
            energy_2_at_1=e21,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_mean=(lambda_1 + lambda_2) / 2,
            vertical_energy_1_to_2=(e21 - e11) * Ha_to_eV,
            vertical_energy_2_to_1=(e12 - e22) * Ha_to_eV,
            adiabatic_energy=(e22 - e11) * Ha_to_eV,
            adiabatic_free_energy=adiabatic_free_energy,
            geometry_rmsd=kabsch_rmsd(molecule_1, molecule_2),
            connectivity_change=_connectivity_change(molecule_1, molecule_2),
            **kwargs,
        )


def kabsch_rmsd(molecule_1: Molecule, molecule_2: Molecule) -> float:
    """
    RMSD between two geometries with the same atom order, after optimal superposition.

    Parameters
    ----------
    molecule_1, molecule_2 : .Molecule
        The two geometries.

    Returns
    -------
    float
        The RMSD in Angstrom.
    """
    if [str(s) for s in molecule_1.species] != [str(s) for s in molecule_2.species]:
        raise ValueError(
            "The two molecules must have the same atoms in the same order."
        )
    coords_1 = molecule_1.cart_coords - molecule_1.cart_coords.mean(axis=0)
    coords_2 = molecule_2.cart_coords - molecule_2.cart_coords.mean(axis=0)
    u, _, vt = np.linalg.svd(coords_1.T @ coords_2)
    sign = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    diff = coords_1 @ rotation - coords_2
    return float(np.sqrt((diff**2).sum(axis=1).mean()))


def _connectivity_change(molecule_1: Molecule, molecule_2: Molecule) -> str | None:
    try:
        from pymatgen.io.qchem.outputs import check_for_structure_changes

        return check_for_structure_changes(molecule_1, molecule_2)
    except ImportError:
        logger.warning("openbabel is not available; skipping the connectivity check.")
        return None
