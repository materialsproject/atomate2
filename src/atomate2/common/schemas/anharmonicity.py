"""Schemas for anharmonicity quantification"""

import logging
from typing import Optional, Union, Self

from emmet.core.math import Matrix3D
from emmet.core.structure import StructureMetadata
from pydantic import Field
from pymatgen.core.structure import Structure
from atomate2.common.schemas.phonons import PhononBSDOSDoc

logger = logging.getLogger(__name__)

class AnharmonicityDoc(StructureMetadata, extra="allow"):
    """Collection to store data from anharmonicity workflow"""

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

    sigma_dict: Optional[dict[str, Union[list, float]]] = Field(
        None,
        description="Dictionary with all computed sigma^A forms"
    )

    @classmethod
    def store_data(
        cls,
        sigma_dict: dict[str, Union[list, float]],
        phonon_doc: PhononBSDOSDoc,
        one_shot: bool
    ) -> Self:
        """
        Generates the collection of data for the anharmonicity workflow

        Parameters
        ----------
        sigma_dict: dict[str, Union[list, float]]
            Dictionary of computed sigma^A values.
            Possible contents are full, one-shot, atom-resolved, and
            mode-resolved.
        phonon_doc: PhononBSDOSDoc
            Document with data from phonon workflow
        one_shot: bool
            True if one shot approximation was found, false otherwise
        """
        return cls.from_structure(
            structure=phonon_doc.structure,
            meta_structure=phonon_doc.structure,
            sigma_dict=sigma_dict,
            phonon_doc=phonon_doc,
            supercell_matrix=phonon_doc.supercell_matrix,
            primitive_matrix=phonon_doc.primitive_matrix,
            one_shot=one_shot,
        )
