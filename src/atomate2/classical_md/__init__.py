from openff.toolkit.topology.molecule import Molecule
from openff.interchange import Interchange
from openff.toolkit.topology import Topology

from pydantic import parse


def openff_mol_as_monty_dict(self):
    mol_dict = self.to_dict()
    mol_dict["@module"] = "openff.toolkit.topology"
    mol_dict["@class"] = "Molecule"
    return mol_dict


Molecule.as_dict = openff_mol_as_monty_dict


def openff_topology_as_monty_dict(self):
    mol_dict = self.to_dict()
    mol_dict["@module"] = "openff.toolkit.topology"
    mol_dict["@class"] = "Topology"
    return mol_dict


Topology.as_dict = openff_topology_as_monty_dict


def openff_interchange_as_monty_dict(self):
    mol_dict = self.dict()
    mol_dict["@module"] = "openff.interchange"
    mol_dict["@class"] = "Interchange"
    return mol_dict


def openff_interchange_from_monty_dict(cls, d):
    d = d.copy()
    d.pop("@module")
    d.pop("@class")
    return cls(**d)


Interchange.as_dict = openff_interchange_as_monty_dict
Interchange.from_dict = openff_interchange_from_monty_dict


from openff.units import Quantity


def openff_quantity_as_monty_dict(self):
    q_tuple = self.to_tuple()
    q_dict = {"magnitude": q_tuple[0], "unit": q_tuple[1]}
    q_dict["@module"] = "openff.units"
    q_dict["@class"] = "Quantity"
    return q_dict


def openff_quantity_from_monty_dict(cls, d):
    d = d.copy()
    d.pop("@module")
    d.pop("@class")
    q_tuple = (d["magnitude"], d["unit"])
    return cls.from_tuple(q_tuple)


Quantity.as_dict = openff_quantity_as_monty_dict
Quantity.from_dict = openff_quantity_from_monty_dict
