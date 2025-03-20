"""Schemas for anharmonicity quantification."""

import logging
from typing import Any, Optional

from emmet.core.math import Matrix3D
from emmet.core.structure import StructureMetadata
from pydantic import Field
from pymatgen.core.structure import Structure

from atomate2.common.schemas.phonons import PhononBSDOSDoc

logger = logging.getLogger(__name__)


class AnharmonicityDoc(StructureMetadata):
    """Collection to store data from anharmonicity workflow."""

    phonon_doc: Optional[PhononBSDOSDoc] = Field(
        None, description="Collection of data from phonon part of the workflow"
    )

    supercell_matrix: Matrix3D = Field("Matrix describing the supercell")

    structure: Optional[Structure] = Field(
        None, description="Structure of Materials Project."
    )

    primitive_matrix: Matrix3D = Field(
        "matrix describing relationship to primitive cell"
    )

    sigma_dict: Optional[dict[str, Any]] = Field(
        None, description="Dictionary with all computed sigma^A forms"
    )

    parameters_dict: Optional[dict] = Field(
        None, description="Parameters used for anharmonicity quantification"
    )

    @classmethod
    def from_phonon_doc_sigma(
        cls,
        sigma_dict: dict[str, Any],
        phonon_doc: PhononBSDOSDoc,
        one_shot: bool,
        temp: float,
        n_samples: int,
        seed: Optional[int],
    ) -> "AnharmonicityDoc":
        """
        Generate the collection of data for the anharmonicity workflow.

        Parameters
        ----------
        sigma_dict: dict[str, Any]
            Dictionary of computed sigma^A values.
            Possible contents are full, one-shot, atom-resolved, and
            mode-resolved.
        phonon_doc: PhononBSDOSDoc
            Document with data from phonon workflow
        one_shot: bool
            True if one shot approximation was found, false otherwise
        temp: float
            Temperature (in K) to displace structures at
        n_samples: int
            How many displaced structures to sample
        seed: Optional[int]
            What random seed to use for displacing structures

        Returns
        -------
        AnharmonicityDoc
            Document with details about anharmonicity and phonon
            workflow runs
        """
        return cls.from_structure(
            structure=phonon_doc.structure,
            meta_structure=phonon_doc.structure,
            sigma_dict=sigma_dict,
            phonon_doc=phonon_doc,
            supercell_matrix=phonon_doc.supercell_matrix,
            primitive_matrix=phonon_doc.primitive_matrix,
            parameters_dict={
                "one-shot": one_shot,
                "temp": temp,
                "num_samples": n_samples,
                "seed": seed,
            },
        )
