"""Jobs used in the calculation of elastic tensors."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from jobflow import Flow, Response, job
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.analysis.elasticity import Deformation, Strain, Stress
from pymatgen.core import SymmOp
from pymatgen.core.structure import Structure
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

from atomate2.common.analysis.elastic import get_default_strain_states
from atomate2.common.schemas.elastic import ElasticDocument
from atomate2.common.schemas.math import Matrix3D
from atomate2.settings import settings
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.base import VaspInputSetGenerator
from atomate2.vasp.sets.elastic import ElasticDeformationSetGenerator

logger = logging.getLogger(__name__)

__all__ = [
    "ElasticRelaxMaker",
    "generate_elastic_deformations",
    "run_elastic_deformations",
    "fit_elastic_tensor",
]


@dataclass
class ElasticRelaxMaker(BaseVaspMaker):
    """
    Maker to perform an elastic relaxation.

    This is a tight relaxation where only the atom positions are allowed to relax.
    """

    name = "elastic relax"
    input_set_generator: VaspInputSetGenerator = field(
        default_factory=ElasticDeformationSetGenerator
    )


@job
def generate_elastic_deformations(
    structure: Structure,
    order: int = 2,
    strain_states: List[Tuple[int, int, int, int, int, int]] = None,
    strain_magnitudes: Union[List[float], List[List[float]]] = None,
    conventional: bool = False,
    symprec: float = settings.SYMPREC,
    sym_reduce: bool = True,
):
    """
    Generate elastic deformations.

    Parameters
    ----------
    structure
        A pymatgen structure object.
    order
        Order of the tensor expansion to be determined. Can be either 2 or 3.
    strain_states
        List of Voigt-notation strains, e.g. ``[(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0),
        etc]``.
    strain_magnitudes
        A list of strain magnitudes to multiply by for each strain state, e.g. ``[-0.01,
        -0.005, 0.005, 0.01]``. Alternatively, a list of lists can be specified, where
        each inner list corresponds to a specific strain state.
    conventional
        Whether to transform the structure into the conventional cell.
    symprec
        Symmetry precision.
    sym_reduce
        Whether to reduce the number of deformations using symmetry.

    Returns
    -------
    Dict[str, Any]
        A dictionary with the keys:

        - "deformations": containing a list of deformations.
        - "symmetry_ops": containing a list of symmetry operations or None if
          symmetry_reduce is False.
    """
    if conventional:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        structure = sga.get_conventional_standard_structure()

    if strain_states is None:
        strain_states = get_default_strain_states(order)

    if strain_magnitudes is None:
        strain_magnitudes = np.linspace(-0.01, 0.01, 5 + (order - 2) * 2)

    if np.array(strain_magnitudes).ndim == 1:
        strain_magnitudes = [strain_magnitudes] * len(strain_states)  # type: ignore

    strains = []
    for state, magnitudes in zip(strain_states, strain_magnitudes):
        strains.extend([Strain.from_voigt(m * np.array(state)) for m in magnitudes])  # type: ignore

    # remove zero strains
    strains = [strain for strain in strains if (abs(strain) > 1e-10).any()]

    if np.linalg.matrix_rank([strain.voigt for strain in strains]) < 6:
        # TODO: check for sufficiency of input for nth order
        raise ValueError("strain list is insufficient to fit an elastic tensor")

    deformations = [s.get_deformation_matrix() for s in strains]

    symmetry_operations = None
    if sym_reduce:
        deformation_mapping = symmetry_reduce(deformations, structure, symprec=symprec)
        logger.info(
            f"Using symmetry to reduce number of deformations from {len(deformations)} "
            f"to {len(list(deformation_mapping.keys()))}"
        )
        deformations = list(deformation_mapping.keys())
        symmetry_operations = list(deformation_mapping.values())

    return {"deformations": deformations, "symmetry_ops": symmetry_operations}


@job
def run_elastic_deformations(
    structure: Structure,
    deformations: List[Deformation],
    symmetry_ops: List[SymmOp] = None,
    prev_vasp_dir: Union[str, Path] = None,
    elastic_relax_maker: BaseVaspMaker = None,
):
    """
    Run elastic deformations.

    Note, this job will replace itself with N relaxation calculations, where N is
    the number of deformations.

    Parameters
    ----------
    structure
        A pymatgen structure.
    deformations
        The deformations to apply.
    symmetry_ops
        A list of symmetry operations (must be same number as deformations).
    prev_vasp_dir
        A previous VASP directory to use for copying VASP outputs.
    elastic_relax_maker
        A VaspMaker to use to generate the elastic relaxation jobs.
    """
    if elastic_relax_maker is None:
        elastic_relax_maker = ElasticRelaxMaker()
    if symmetry_ops is not None and len(symmetry_ops) != len(deformations):
        raise ValueError(
            "Number of deformations and lists of symmetry operations must be equal."
        )

    relaxations = []
    outputs = []
    for i, deformation in enumerate(deformations):
        # deform the structure
        dst = DeformStructureTransformation(deformation=deformation)
        ts = TransformedStructure(structure, transformations=[dst])
        deformed_structure = ts.final_structure

        # write details of the transformation to the transformations.json file
        # this file will automatically get added to the task document and allow
        # the elastic builder to function
        elastic_relax_maker.write_additional_data["transformations.json"] = ts

        # create the job
        relax_job = elastic_relax_maker.make(
            deformed_structure, prev_vasp_dir=prev_vasp_dir
        )
        relax_job.name += f" {i + 1}/{len(deformations)}"
        relaxations.append(relax_job)

        # extract the outputs we want
        output = {
            "stress": relax_job.output.output.stress,
            "deformation": deformation,
        }

        if symmetry_ops is not None:
            output["symmetry_ops"] = symmetry_ops[i]

        outputs.append(output)

    relax_flow = Flow(relaxations, outputs)
    return Response(replace=relax_flow)


@job(output_schema=ElasticDocument)
def fit_elastic_tensor(
    structure: Structure,
    deformation_data: List[dict],
    equilibrium_stress: Optional[Matrix3D] = None,
    order: int = 2,
    fitting_method: str = "finite_difference",
):
    """
    Analyze stress/strain data to fit the elastic tensor and related properties.

    Parameters
    ----------
    structure
        A pymatgen structure.
    deformation_data
        The deformation data, as a list of dictionaries, each containing the keys
        "stress", "deformation", and (optionally) "symmetry_ops".
    equilibrium_stress
        The equilibrium stress of the (relaxed) structure, if known.
    order
        Order of the tensor expansion to be fitted. Can be either 2 or 3.
    fitting_method
        The method used to fit the elastic tensor. See pymatgen for more details on the
        methods themselves. The options are:
        - "finite_difference" (note this is required if fitting a 3rd order tensor)
        - "independent"
        - "pseudoinverse"
    """
    stresses = []
    deformations = []
    for data in deformation_data:

        # stress could be none if the deformation calculation failed
        if data["stress"] is None:
            continue

        stress = Stress(data["stress"])
        deformation = Deformation(data["deformation"])

        stresses.append(stress)
        deformations.append(deformation)

        # add derived stresses and strains if symmetry operations are present
        for symmop in data.get("symmetry_ops", []):
            stresses.append(stress.transform(symmop))
            deformations.append(deformation.transform(symmop))

    logger.info("Analyzing stress/strain data")

    elastic_doc = ElasticDocument.from_stresses(
        structure,
        stresses,
        deformations,
        fitting_method,
        order,
        equilibrium_stress=equilibrium_stress,
    )
    return elastic_doc
