from monty.json import MSONable
from dataclasses import dataclass
from pydantic import BaseModel, PositiveInt, confloat, constr, field_validator
from typing import List, Optional, Union
from pathlib import Path
import pymatgen
import openff.toolkit as tk

from pymatgen.analysis.graphs import MoleculeGraph


def xyz_to_molecule(
    mol_geometry: Union[pymatgen.core.Molecule, str, Path]
) -> pymatgen.core.Molecule:
    """
    Convert a XYZ file to a Pymatgen.Molecule.

    Accepts a str or pathlib.Path file that can be parsed for xyz coordinates from OpenBabel and
    returns a Pymatgen.Molecule. If a Pymatgen.Molecule is passed in, it is returned unchanged.

    Args:
        mol_geometry:

    Returns:

    """
    if isinstance(mol_geometry, (str, Path)):
        mol_geometry = pymatgen.core.Molecule.from_file(str(mol_geometry))
    return mol_geometry


class Geometry(BaseModel):
    """
    A geometry schema to be used for input to OpenMMSolutionGen.
    """

    xyz: Union[pymatgen.core.Molecule, str, Path]

    @field_validator("xyz")
    def xyz_is_valid(cls, xyz):
        """check that xyz generates a valid molecule"""
        try:
            xyz_to_molecule(xyz)
        except Exception:
            raise ValueError(f"Invalid xyz file or molecule: {xyz}")
        return xyz_to_molecule(xyz)


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


@dataclass
class InterchangeSpec(MSONable):
    molecule_specs: List[MoleculeSpec]
    forcefield: str
    water_model: str
    # atom_types: List[int]  # could be derived
    # atom_resnames: List[str]  # could be derived


@dataclass
class SetContents(MSONable):
    """
    The molecular contents of an OpenMMSet
    """

    molecule_specs: List[MoleculeSpec]
    force_fields: List[str]
    partial_charge_methods: List[str]
    atom_types: List[int]
    atom_resnames: List[str]  # TODO: include residue types information
    # molecule_resids: Dict[str, List[int]]
