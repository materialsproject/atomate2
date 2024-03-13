from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List

import pymatgen
from emmet.core.vasp.task_valid import TaskState
from monty.json import MSONable
from openff import toolkit as tk
from openff.interchange import Interchange
from pydantic import BaseModel, field_validator, confloat, PositiveInt, Field
from pydantic.dataclasses import dataclass
from pymatgen.analysis.graphs import MoleculeGraph


class Geometry(BaseModel):
    """
    A geometry schema to be used for input to OpenMMSolutionGen.
    """

    xyz: Union[pymatgen.core.Molecule, str, Path]

    @field_validator("xyz")
    def xyz_is_valid(cls, xyz):
        """check that xyz generates a valid molecule"""
        if isinstance(xyz, (str, Path)):
            try:
                xyz = pymatgen.core.Molecule.from_file(str(xyz))
            except Exception:
                raise ValueError(f"Invalid xyz file or molecule: {xyz}")
        return xyz


class InputMoleculeSpec(BaseModel):
    """
    A molecule schema to be used for input to OpenMMSolutionGen.
    """

    smile: str
    count: int
    name: Optional[str] = None
    charge_scaling: Optional[confloat(ge=0.1, le=10)] = 1.0  # type: ignore
    geometries: Optional[List[Geometry]] = None
    partial_charges: Optional[List[float]] = None
    charge_method: Optional[str] = None
    max_conformers: PositiveInt = 1

    class Config:
        # needed to allow for np.ndarray
        arbitrary_types_allowed = True

    @field_validator("smile")
    @classmethod
    def smile_is_valid(cls, smile):
        """check that smile generates a valid molecule"""
        try:
            tk.Molecule.from_smiles(smile, allow_undefined_stereo=True)
        except Exception as smile_error:
            raise ValueError(
                f"Invalid SMILES string: {smile} "
                f"OpenFF Toolkit returned the following "
                f"error: {smile_error}"
            )
        return smile

    @field_validator("force_field", mode="before")
    @classmethod
    def lower_case_ff(cls, force_field):
        """check that force_field is valid"""
        return force_field.lower()

    @field_validator("name")
    @classmethod
    def set_name(cls, name, values):
        """assign name if not provided"""
        if name is None:
            return values.get("smile")
        return name

    @field_validator("geometries", mode="before")
    @classmethod
    def convert_xyz_to_geometry(cls, geometries, values):
        """convert xyz to Geometry"""
        if geometries is not None:
            geometries = [Geometry(xyz=xyz) for xyz in geometries]
            # assert xyz lengths are the same
            n_atoms = tk.Molecule.from_smiles(
                values["smile"], allow_undefined_stereo=True
            ).n_atoms
            if not all([len(geometry.xyz) == n_atoms for geometry in geometries]):
                raise ValueError(
                    "All geometries must have the same number of atoms as the molecule"
                    " defined in the SMILES string."
                )
            return geometries
        return geometries

    @field_validator("partial_charges", mode="before")
    @classmethod
    def check_geometry_is_set(cls, partial_charges, values):
        """check that geometries is set if partial_charges is set"""
        if partial_charges is not None:
            geometries = values.get("geometries")
            if geometries is None:
                raise ValueError("geometries must be set if partial_charges is set")
            if not len(partial_charges) == len(geometries[0].xyz):
                raise ValueError(
                    "partial_charges must be the same length as all geometries"
                )
        return list(partial_charges)

    @field_validator("charge_method", mode="before")
    @classmethod
    def set_custom_charge_method(cls, charge_method, values):
        """label partial charge method if partial_charges is set, defaults to 'custom'"""
        # if partial_charge_label is not None:
        #     if values.get("partial_charges") is None:
        #         raise ValueError(
        #             "partial_charges must be set if partial_charge_label is set"
        #         )
        #     return partial_charge_label
        # else:
        if values.get("partial_charges") is not None and charge_method is None:
            return "custom"
        return charge_method


@dataclass
class MoleculeSpec(MSONable):
    """
    A molecule schema to be output by OpenMMGenerators.
    """

    name: str
    count: int
    smile: str
    formal_charge: int
    charge_method: str
    molgraph: MoleculeGraph
    # TODO: include openff mol?


class ClassicalMDTaskDocument(BaseModel, extra="allow"):
    """Definition of the OpenMM task document."""

    tags: Optional[List[str]] = Field(
        [], title="tag", description="Metadata tagged to a given task."
    )
    dir_name: Optional[str] = Field(
        None, description="The directory for this VASP task"
    )
    state: Optional[TaskState] = Field(None, description="State of this calculation")

    calcs_reversed: Optional[List] = Field(
        None,
        title="Calcs reversed data",
        description="Detailed data for each VASP calculation contributing to the task document.",
    )

    interchange: Optional[Interchange] = Field(
        None, description="Final output structure from the task"
    )

    molecule_specs: Optional[List[MoleculeSpec]] = Field(
        None, description="Molecules within the box."
    )

    forcefield: Optional[str | List[str]] = Field(None, description="forcefield")

    task_type: Optional[str] = Field(None, description="The type of calculation.")

    # task_label: Optional[str] = Field(None, description="A description of the task")
    # TODO: where does task_label get added
    # additional_json

    last_updated: Optional[datetime] = Field(
        None,
        description="Timestamp for the most recent calculation for this task document",
    )
