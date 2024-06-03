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

# Is the extra="allow" needed? What are pros/cons to including it?
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
    
    @classmethod
    def store_data(
        cls,
        sigma_A: float,
        phonon_doc: PhononBSDOSDoc
    ) -> Self:
        """
        Generates the collection of data for the anharmonicity workflow

        Parameters
        ----------
        sigma_A: float
            Float with sigma_A value to be stored
        phonon_doc: PhononBSDOSDoc
            Document with data from phonon workflow
        """
        return cls.from_structure(
            structure=phonon_doc.structure,
            meta_structure=phonon_doc.structure,
            sigma_A=sigma_A,
            phonon_doc=phonon_doc
        )
