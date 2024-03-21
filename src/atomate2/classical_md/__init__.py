"""Module for classical md workflows."""

from openff.interchange import Interchange
from openff.toolkit.topology import Topology
from openff.toolkit.topology.molecule import Molecule
from openff.units import Quantity


def openff_mol_as_monty_dict(self: Molecule) -> dict:
    """Convert a Molecule to a monty dictionary."""
    mol_dict = self.to_dict()
    mol_dict["@module"] = "openff.toolkit.topology"
    mol_dict["@class"] = "Molecule"
    return mol_dict


Molecule.as_dict = openff_mol_as_monty_dict


def openff_topology_as_monty_dict(self: Topology) -> dict:
    """Convert a Topology to a monty dictionary."""
    top_dict = self.to_dict()
    top_dict["@module"] = "openff.toolkit.topology"
    top_dict["@class"] = "Topology"
    return top_dict


Topology.as_dict = openff_topology_as_monty_dict


def openff_interchange_as_monty_dict(self: Interchange) -> dict:
    """Convert an Interchange to a monty dictionary."""
    int_dict = self.dict()
    int_dict["@module"] = "openff.interchange"
    int_dict["@class"] = "Interchange"
    return int_dict


def openff_interchange_from_monty_dict(cls: type[Interchange], d: dict) -> Interchange:
    """Construct an Interchange from a monty dictionary."""
    d = d.copy()
    d.pop("@module", None)
    d.pop("@class", None)
    return cls(**d)


Interchange.as_dict = openff_interchange_as_monty_dict
Interchange.from_dict = classmethod(openff_interchange_from_monty_dict)


def openff_quantity_as_monty_dict(self: Quantity) -> dict:
    """Convert a Quantity to a monty dictionary."""
    q_tuple = self.to_tuple()
    q_dict = {"magnitude": q_tuple[0], "unit": q_tuple[1]}
    q_dict["@module"] = "openff.units"
    q_dict["@class"] = "Quantity"
    return q_dict


def openff_quantity_from_monty_dict(cls: type[Quantity], d: dict) -> Quantity:
    """Construct a Quantity from a monty dictionary."""
    d = d.copy()
    d.pop("@module", None)
    d.pop("@class", None)
    q_tuple = (d["magnitude"], d["unit"])
    return cls.from_tuple(q_tuple)


Quantity.as_dict = openff_quantity_as_monty_dict
Quantity.from_dict = classmethod(openff_quantity_from_monty_dict)
