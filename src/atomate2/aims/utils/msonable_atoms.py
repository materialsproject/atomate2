"""A MSONable ASE Atoms Object."""

from typing import Any, Dict

import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointDFTCalculator
from monty.json import MontyDecoder, MSONable
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

ASE_ADAPTOR = AseAtomsAdaptor()


class MSONableAtoms(Atoms, MSONable):
    """A MSONable ASE atoms object."""

    def as_dict(self) -> Dict[str, Any]:
        """Represent an ASE Atoms object as a dict.

        Returns
        -------
        The dictionary representation of the Atoms object
        """
        d = {"@module": self.__class__.__module__, "@class": self.__class__.__name__}

        for key, val in self.todict().items():
            d[key] = val

        if self.calc:
            d["calculated_results"] = self.calc.results

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """Create the Atoms object from the dictionary.

        Parameters
        ----------
        d: Dict[str, Any]
            The dictionary representation of the Atoms object
        """
        decoded = {
            k: MontyDecoder().process_decoded(v)
            for k, v in d.items()
            if not k.startswith("@")
        }
        calculated_results = decoded.pop("calculated_results", None)

        atoms = Atoms.fromdict(decoded)

        calculator = SinglePointDFTCalculator(atoms)
        calculator.results = calculated_results

        return cls(atoms, calculator=calculator)

    @classmethod
    def from_pymatgen(cls, structure: Structure | Molecule):
        """Create an Atoms object from a pymatgen object."""
        return ASE_ADAPTOR.get_atoms(structure)

    @property
    def structure(self) -> Structure:
        """The pymatgen Structure of the Atoms object."""
        return ASE_ADAPTOR.get_structure(self)

    @property
    def pymatgen(self) -> Structure | Molecule:
        """The pymatgen Structure or Molecule of the Atoms object."""
        if np.any(self.pbc):
            return ASE_ADAPTOR.get_structure(self)

        return ASE_ADAPTOR.get_molecule(self)
