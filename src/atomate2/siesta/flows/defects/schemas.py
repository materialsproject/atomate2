"""Schemas for defect calculations."""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pymatgen.core import Structure

from atomate2.siesta.schemas.task import SiestaTaskDoc


class DefectDocument(SiestaTaskDoc):
    """
    Document for defect calculation results.

    Contains formation energy, charge correction, and defect properties.
    """

    defect_type: str = Field(
        description="Type of defect (vacancy, substitutional, interstitial)"
    )

    defect_species: str | None = Field(
        None,
        description=(
            "Species of the defect (added species for substitution, "
            "removed for vacancy)"
        ),
    )

    removed_species: str | None = Field(
        None,
        description="For substitution only: the removed (host) species",
    )

    defect_site: list[float] | None = Field(
        None,
        description="Fractional coordinates of defect site [x, y, z]",
    )

    charge_state: int = Field(
        description="Charge state of the defect (e.g., +2, 0, -1)"
    )

    # Energies
    defect_energy: float = Field(description="Total energy of defect supercell (eV)")

    host_energy: float = Field(
        description="Total energy of pristine host supercell (eV)"
    )

    raw_formation_energy: float = Field(description="Uncorrected formation energy (eV)")

    # Correction
    correction_scheme: str = Field(description="Finite-size correction scheme used")

    correction_energy: float = Field(
        0.0,
        description="Finite-size correction energy (eV)",
    )

    corrected_formation_energy: float = Field(
        description="Formation energy with correction (eV)"
    )

    # Chemical potential and Fermi level
    chemical_potential: float = Field(
        0.0,
        description=(
            "Net chemical potential contribution (eV): "
            "μ for vacancy/interstitial, Δμ for substitution"
        ),
    )

    mu_removed: float = Field(
        0.0,
        description=(
            "Chemical potential of removed species (eV), "
            "for substitution/vacancy"
        ),
    )

    mu_added: float = Field(
        0.0,
        description=(
            "Chemical potential of added species (eV), "
            "for substitution/interstitial"
        ),
    )

    fermi_level: float = Field(
        0.0,
        description="Fermi level position (eV, relative to VBM)",
    )

    # Supercell info
    supercell_matrix: list[list[int]] = Field(
        description="Supercell transformation matrix"
    )

    supercell_natoms: int = Field(description="Number of atoms in supercell")

    # Dielectric properties
    epsilon_static: float | None = Field(
        None,
        description="Static dielectric constant (for corrections)",
    )

    # Host properties
    host_bandgap: float | None = Field(
        None,
        description="Bandgap of the pristine host material (eV)",
    )

    # Structures
    defect_structure: Structure = Field(description="Relaxed defect structure")

    host_structure: Structure = Field(description="Pristine host structure")

    # Additional analysis
    correction_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional correction scheme metadata",
    )

    @classmethod
    def from_defect_calc(
        cls,
        task_doc: SiestaTaskDoc,
        defect_type: str,
        charge_state: int,
        correction_energy: float,
        correction_scheme: str,
        host_energy: float,
        host_structure: Structure,
        chemical_potential: float = 0.0,
        fermi_level: float = 0.0,
        **kwargs,
    ) -> DefectDocument:
        """
        Create DefectDocument from SiestaTaskDoc and correction results.

        Parameters
        ----------
        task_doc : SiestaTaskDoc
            SiestaTaskDoc from defect calculation
        defect_type : str
            Type of defect
        charge_state : int
            Charge state
        correction_energy : float
            Finite-size correction (eV)
        correction_scheme : str
            Correction scheme name
        host_energy : float
            Host energy (eV)
        host_structure : Structure
            Host structure
        chemical_potential : float
            Chemical potential (eV)
        fermi_level : float
            Fermi level (eV)
        **kwargs
            Additional fields

        Returns
        -------
        DefectDocument
            Defect calculation document
        """
        defect_energy = task_doc.output.energy
        structure = task_doc.output.structure

        if defect_energy is None:
            raise ValueError(
                "Defect task_doc has no energy (output.energy is None). "
                "The SIESTA calculation likely did not converge. "
                "Check the .out file for SCF convergence or ABNORMAL_TERMINATION."
            )

        # Calculate raw formation energy (includes chemical potential)
        raw_formation_energy = defect_energy - host_energy + chemical_potential

        # Apply correction
        corrected_formation_energy = (
            raw_formation_energy + correction_energy + charge_state * fermi_level
        )

        # Get supercell info from structure
        supercell_natoms = len(structure)

        return cls(
            **task_doc.model_dump(),
            defect_type=defect_type,
            charge_state=charge_state,
            defect_energy=defect_energy,
            host_energy=host_energy,
            raw_formation_energy=raw_formation_energy,
            correction_scheme=correction_scheme,
            correction_energy=correction_energy,
            corrected_formation_energy=corrected_formation_energy,
            chemical_potential=chemical_potential,
            fermi_level=fermi_level,
            supercell_natoms=supercell_natoms,
            defect_structure=structure,
            host_structure=host_structure,
            **kwargs,
        )
