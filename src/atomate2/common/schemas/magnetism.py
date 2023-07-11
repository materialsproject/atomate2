"""Schemas for collinear magnetic ordering flows."""
from __future__ import annotations

from pydantic import BaseModel, Field
from pymatgen.analysis.magnetism.analyzer import Ordering
from pymatgen.core.structure import Structure


class MagneticOrderingInput(BaseModel):
    """
    Defines the input strtucture/ordering for a magnetic ordering calculation. This is embedded
    in the MagneticOrderingOutput and MagneticOrderingRelaxation documents.
    """

    structure: Structure = Field(None, description="Input structure")
    ordering: Ordering = Field(
        None,
        description=(
            "The magnetic ordering of the input structure, "
            "as defined in pymatgen.analysis.magnetism.analyzer."
        ),
    )
    symmetry: str = Field(None, description="Detected space group symbol.")


class MagneticOrderingRelaxation(BaseModel):
    """
    Defines the relaxation information for a magnetic ordering calculation. This is
    embedded within the MagneticOrderingOutput.
    """

    uuid: str = Field(None, description="Unique ID of the calculation.")
    dir_name: str = Field(None, description="Directory of the calculation.")
    input: MagneticOrderingInput = Field(
        None, description="Input ordering information."
    )
    structure: Structure = Field(
        None, description="Final structure from the calculation."
    )
    symmetry_changed: bool = Field(None, description="Whether or not symmetry changed.")
    ordering_changed: bool = Field(
        None,
        description=(
            "Specifies whether or not the magnetic ordering changed during the"
            " calculation."
        ),
    )
    ordering: Ordering = Field(None, description="Final ordering from the calculation.")
    symmetry: str = Field(None, description="Detected space group symbol.")
    energy: float = Field(None, description="Final energy result from the calculation.")
    energy_per_atom: float = Field(None, description="Final energy per atom.")

    @classmethod
    def from_task_document(cls, task_document, uuid=None) -> MagneticOrderingOutput:
        """
        Construct a MagneticOrderingRelaxation output doc from a task document. This is
        to be implemented for the DFT code of choice.
        """
        raise NotImplementedError


class MagneticOrderingOutput(BaseModel):
    """
    Defines the output for a *static* magnetic ordering calculation. This is used
    within the construction of the MagneticOrderingDocument. If a relaxation was
    performed, this information will be stored within the relax_output field.
    """

    uuid: str = Field(None, description="Unique ID of the calculation.")
    dir_name: str = Field(None, description="Directory of the calculation.")
    input: MagneticOrderingInput = Field(
        None, description="Input ordering information."
    )
    structure: Structure = Field(
        None, description="Final structure from the calculation."
    )
    ordering: Ordering = Field(
        None,
        description=(
            "The magnetic ordering of the output structure, "
            "as defined in pymatgen.analysis.magnetism.analyzer."
        ),
    )
    magmoms: list[float] = Field(None, description="Magnetic moments of the structure.")
    symmetry: str = Field(None, description="Detected space group symbol.")
    energy: float = Field(None, description="Final energy result from the calculation.")
    energy_per_atom: float = Field(None, description="Final energy per atom.")
    total_magnetization: float = Field(
        None,
        description=(
            "Total magnetization as a sum of individual atomic moments in "
            "the calculated unit cell."
        ),
    )
    total_magnetization_per_formula_unit: float = Field(
        None, description="Total magnetization normalized to per formula unit."
    )
    total_magnetization_per_unit_volume: float = Field(
        None, description="Total magnetiation noramlized to per unit volume."
    )
    ordering_changed: bool = Field(
        None,
        description=(
            "Specifies whether or not the magnetic ordering changed during the"
            " calculation."
        ),
    )
    symmetry_changed: bool = Field(
        None,
        description=(
            "Specifies whether or not the symmetry changed during the calculation."
        ),
    )
    energy_above_ground_state_per_atom: float = Field(
        None, description="Energy per atom above the calculated ground state ordering."
    )
    relax_output: MagneticOrderingRelaxation | None = Field(
        None, description="Relaxation output, if relaxation performed."
    )
    energy_diff_relax_static: str | float | None = Field(
        None,
        description=(
            "Difference in energy between relaxation and final static calculation, if"
            " relaxation performed (useful for benchmarking). Specifically, this is"
            " calculated as energy[static] - energy[relax]."
        ),
    )

    @classmethod
    def from_task_document(cls, task_document, uuid=None) -> MagneticOrderingOutput:
        """
        Construct a MagnetismOutput from a task document. This is to be implemented for
        the DFT code of choice.
        """
        raise NotImplementedError


class MagneticOrderingDocument(BaseModel):
    """
    Final document containing information about calculated magnetic orderings of a
    structure, including description of the ground state ordering.

    This document is returned by the MagneticOrderingsBuilder corresponding to your DFT
    code.
    """

    formula: str = Field(
        None,
        description="Formula taken from pymatgen.core.structure.Structure.formula.",
    )
    formula_pretty: str = Field(
        None,
        description="Cleaned representation of the formula",
    )
    parent_structure: Structure = Field(
        None,
        description=(
            "The parent structure from which individual magnetic "
            "orderings are generated."
        ),
    )
    outputs: list[MagneticOrderingOutput] = Field(
        None,
        description="All magnetic ordering calculation results for this structure.",
    )
    ground_state_uuid: str = Field(
        None, description="UUID of the ground state ordering."
    )
    ground_state_structure: Structure = Field(
        None, description="Ground state structure."
    )
    ground_state_ordering: Ordering = Field(None, description="Ground state ordering.")
    ground_state_energy: float = Field(None, description="Ground state energy.")
    ground_state_energy_per_atom: float = Field(
        None, description="Ground state energy."
    )

    @classmethod
    def from_outputs(
        cls,
        outputs: list[MagneticOrderingOutput],
        parent_structure: Structure,
    ) -> MagneticOrderingDocument:
        """
        Construct a MagneticOrderingDocument from a list of MagneticOrderingOutput docs.
        This is general and should not need to be implemented for a specific DFT code.
        """
        formula = outputs[0].structure.formula
        formula_pretty = outputs[0].structure.composition.reduced_formula

        ground_state = min(outputs, key=lambda struct: struct.energy_per_atom)

        return cls(
            formula=formula,
            formula_pretty=formula_pretty,
            parent_structure=parent_structure,
            outputs=outputs,
            ground_state_uuid=ground_state.uuid,
            ground_state_structure=ground_state.structure,
            ground_state_ordering=ground_state.ordering,
            ground_state_energy=ground_state.energy,
            ground_state_energy_per_atom=ground_state.energy_per_atom,
        )
