"""Schemas for anharmonicity quantification"""

import copy
import logging
from pathlib import Path
from typing import Optional, Union, Self

import numpy as np
from emmet.core.math import Matrix3D
from emmet.core.structure import StructureMetadata
from monty.json import MSONable
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from phonopy.units import VaspToTHz
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.io.phonopy import (
    get_ph_bs_symm_line,
    get_ph_dos,
    get_phonopy_structure,
    get_pmg_structure,
)
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.symmetry.kpath import KPathSeek

from atomate2.aims.utils.units import omegaToTHz

from atomate2.common.schemas.phonons import (
    PhononBSDOSDoc,
    PhononUUIDs,
    PhononComputationalSettings,
    PhononJobDirs,
    ForceConstants
)

logger = logging.getLogger(__name__)

class AnharmonicityDoc(StructureMetadata, extra="allow"):
    """Collection to store data from anharmonicity workflow"""

    sigma_A: Optional[float] = Field(
        None,
        description="Degree of anharmonicity for the structure"
    )

    phonon_doc: Optional[PhononBSDOSDoc] = Field(
        None,
        description="Collection of data from phonon part of the workflow"
    )

    supercell_matrix: Matrix3D = Field("Matrix describing the supercell")

    structure: Optional[Structure] = Field(
        None, description="Structure of Materials Project."
    )

    primitive_matrix: Matrix3D = Field(
        "matrix describing relationship to primitive cell"
    )

    one_shot: Optional[bool] = Field(
        None, 
        description="Whether or not the one shot approximation was found"
    )

    atom_resolved_sigma_A: list[tuple[str, float]] | None = Field(
        None,
        description="Sigma A values for each mode. Each outer list represents a different atom type. In each"
        "tuple, the string is the atomic symbol and the second is sigma^A resolved to that atom."
    )

    mode_resolved_sigma_A: list[tuple[float, float]] | None = Field(
        None,
        description="Sigma A values for each mode. Each outer list represents a different mode. In each"
        "tuple, the first float is the mode frequency (THz) and the second is sigma^A resolved to that mode."
    )
    
    @classmethod
    def store_data(
        cls,
        sigma_A: float,
        sigma_A_by_atom: list[tuple[str, float]] | None,
        sigma_A_by_mode: list[tuple[float, float]] | None,
        phonon_doc: PhononBSDOSDoc,
        one_shot: bool
    ) -> Self:
        """
        Generates the collection of data for the anharmonicity workflow

        Parameters
        ----------
        sigma_A: float
            Float with sigma_A value to be stored
        sigma_A_by_atom: list[tuple[str, float]] | None
            List of atom-resolved sigma^A values. In each tuple, the string is the atom and 
            the float is sigma^A resolved to that atom.
        sigma_A_by_mode: list[tuple[float, float]] | None
            List of mode-resolved sigma^A values. In each tuple, the first float is the mode
            frequency (THz) and the second is sigma^A resolved to that mode.
        phonon_doc: PhononBSDOSDoc
            Document with data from phonon workflow
        one_shot: bool
            True if one shot approximation was found, false otherwise
        """
        return cls.from_structure(
            structure=phonon_doc.structure,
            meta_structure=phonon_doc.structure,
            sigma_A=sigma_A,
            atom_resolved_sigma_A=sigma_A_by_atom,
            mode_resolved_sigma_A=sigma_A_by_mode,
            phonon_doc=phonon_doc,
            supercell_matrix=phonon_doc.supercell_matrix,
            primitive_matrix=phonon_doc.primitive_matrix,
            one_shot=one_shot,
        )
