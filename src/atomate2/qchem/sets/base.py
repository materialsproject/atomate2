"""Module defining base QChem input set and generator."""

import os
from pickletools import optimize
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from monty.io import zopen
from monty.serialization import loadfn
from pkg_resources import resource_filename
from pymatgen.core.structure import Molecule
from pymatgen.io.qchem.inputs import QCInput
#from pymatgen.io.qchem.utils import lower_and_check_unique
from pymatgen.io.core import InputGenerator
from pymatgen.io.qchem.sets import QChemDictSet
from atomate2 import SETTINGS

__author__ = "Alex Ganose, Ryan Kingsbury, Rishabh D Guha"
__copyright__ = "Copyright 2018-2022, The Materials Project"
__version__ = "0.1"

_BASE_QCHEM_SET = loadfn(resource_filename("atomate2.qchem.sets", "BaseQchemSet.yaml"))

__all__ = ["QChemInputGenerator"]

@dataclass
class QchemInputGenerator(InputGenerator):
    """
    A class to generate QChem input sets for different calculations
    """

    def get_input_set( 
        self,
        molecule: Molecule = None,
        prev_dir: Union[str, Path] = None,
        overwrite_inputs: dict = None
    ) -> QChemDictSet:
        """
        Get a QChem Input Set as a dictionary for a molecule

        Parameters
        ----------
        molecule
            A Pymatgen molecule
        prev_dir
            A previous directory to generate the input set from
        overwrite_inputs

        Returns
        -------
        QchemInputSet
            A QChem input set
        """
        molecule, prev_basis, prev_scf, new_geom_opt, overwrite_inputs, nbo_params = self._get_previous(
            molecule, prev_dir
        )

        basis_set_updates = self.get_basis_set_updates(
            molecule,
            prev_basis=prev_basis,
            prev_scf=prev_scf,
            new_geom_opt=new_geom_opt,
            overwrite_inputs=overwrite_inputs,
            nbo_params=nbo_params,
        )

        scf_algorithm_updates = self.get_scf_algorithm_updates(
            molecule,
            prev_basis=prev_basis,
            prev_scf=prev_scf,
            new_geom_opt=new_geom_opt,
            overwrite_inputs=overwrite_inputs,
            nbo_params=nbo_params,
        )

        basis_set = self._basis_set(basis_set_updates)
        scf_algorithm = self._get_scf_algorithm(scf_algorithm_updates)


        return QChemDictSet(
            molecule=molecule,
            basis_set=basis_set,
            scf_algorithm=scf_algorithm,
        )

def get_basis_set_updates(
    self,
    molecule: Molecule,
    prev_basis: str = None,
    prev_scf: str = None,
    new_geom_opt: dict = None,
    nbo_params: dict = None,
    overwrite_inputs: dict = None, 
) -> dict:

    """
    Get updates to chosen basis set for this calculation type.

    Parameters
    ----------
    molecule
        A molecule
    prev_basis
        A basis set from a previous calculation
    prev_scf
        An scf algorithm from a previous calculation
    new_geom_opt
        The new geometry optimization in QChem 6
    nbo_params
        The natural bonding order information
    overwrite_inputs
        A dictionary to overwrite the current inputs

    Returns
    -------
        A dictionary of updates to apply
    """
    return {}

def get_scf_algorithm_updates(
    self,
    molecule: Molecule,
    prev_basis: str = None,
    prev_scf: str = None,
    new_geom_opt: dict = None,
    nbo_params: dict = None,
    overwrite_inputs: dict = None, 
) -> dict:

    """
    Get updates to scf algorithm for this calculation type.

    Parameters
    ----------
    molecule
        A molecule
    prev_basis
        A basis set from a previous calculation
    prev_scf
        An scf algorithm from a previous calculation
    new_geom_opt
        The new geometry optimization in QChem 6
    nbo_params
        The natural bonding order information
    overwrite_inputs
        A dictionary to overwrite the current inputs

    Returns
    -------
        A dictionary of updates to apply
    """
    return {}
