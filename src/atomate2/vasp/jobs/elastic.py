"""Jobs used in the calculation of elastic tensors."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from emmet.core.math import Matrix3D
from jobflow import Flow, Response, job
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.analysis.elasticity import Deformation, Strain, Stress
from pymatgen.core.structure import Structure
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

from atomate2 import SETTINGS
from atomate2.common.analysis.elastic import get_default_strain_states
from atomate2.common.schemas.elastic import ElasticDocument
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.base import VaspInputGenerator
from atomate2.vasp.sets.core import StaticSetGenerator

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

    The input set is for a tight relaxation, where only the atomic positions are
    allowed to relax (ISIF=2). Both the k-point mesh density and convergence parameters
    are stricter than a normal relaxation.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "elastic relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"grid_density": 7000},
            user_incar_settings={
                "IBRION": 2,
                "ISIF": 2,
                "ENCUT": 700,
                "EDIFF": 1e-7,
                "LAECHG": False,
                "EDIFFG": -0.001,
                "LREAL": False,
                "ALGO": "Normal",
                "NSW": 99,
                "LCHARG": False,
            },
        )
    )


@job
def generate_elastic_deformations(
    structure: Structure,
    order: int = 2,
    strain_states: list[tuple[int, int, int, int, int, int]] | None = None,
    strain_magnitudes: list[float] | list[list[float]] | None = None,
    conventional: bool = False,
    symprec: float = SETTINGS.SYMPREC,
    sym_reduce: bool = True,
):
    """
    Generate elastic deformations.

    Parameters
    ----------
    structure : Structure
        A pymatgen structure object.
    order : int
        Order of the tensor expansion to be determined. Can be either 2 or 3.
    strain_states : None or list of tuple of int
        List of Voigt-notation strains, e.g. ``[(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0),
        etc]``.
    strain_magnitudes : None or list of float or list of list of float
        A list of strain magnitudes to multiply by for each strain state, e.g. ``[-0.01,
        -0.005, 0.005, 0.01]``. Alternatively, a list of lists can be specified, where
        each inner list corresponds to a specific strain state.
    conventional : bool
        Whether to transform the structure into the conventional cell.
    symprec : float
        Symmetry precision.
    sym_reduce : bool
        Whether to reduce the number of deformations using symmetry.

    Returns
    -------
    List[Deformation]
        A list of deformations.
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
        strains.extend(
            [Strain.from_voigt(m * np.array(state)) for m in magnitudes]  # type: ignore
        )

    # remove zero strains
    strains = [strain for strain in strains if (abs(strain) > 1e-10).any()]

    if np.linalg.matrix_rank([strain.voigt for strain in strains]) < 6:
        # TODO: check for sufficiency of input for nth order
        raise ValueError("strain list is insufficient to fit an elastic tensor")

    deformations = [s.get_deformation_matrix() for s in strains]

    if sym_reduce:
        deformation_mapping = symmetry_reduce(deformations, structure, symprec=symprec)
        logger.info(
            f"Using symmetry to reduce number of deformations from {len(deformations)} "
            f"to {len(list(deformation_mapping.keys()))}"
        )
        deformations = list(deformation_mapping.keys())

    return deformations


@job
def run_elastic_deformations(
    structure: Structure,
    deformations: list[Deformation],
    prev_vasp_dir: str | Path | None = None,
    elastic_relax_maker: BaseVaspMaker = None,
):
    """
    Run elastic deformations.

    Note, this job will replace itself with N relaxation calculations, where N is
    the number of deformations.

    Parameters
    ----------
    structure : Structure
        A pymatgen structure.
    deformations : list of Deformation
        The deformations to apply.
    prev_vasp_dir : str or Path or None
        A previous VASP directory to use for copying VASP outputs.
    elastic_relax_maker : .BaseVaspMaker
        A VaspMaker to use to generate the elastic relaxation jobs.
    """
    if elastic_relax_maker is None:
        elastic_relax_maker = ElasticRelaxMaker()

    relaxations = []
    outputs = []
    for i, deformation in enumerate(deformations):
        # deform the structure
        dst = DeformStructureTransformation(deformation=deformation)
        ts = TransformedStructure(structure, transformations=[dst])
        deformed_structure = ts.final_structure

        # write details of the transformation to the transformations.json file
        # this file will automatically get added to the task document and allow
        # the elastic builder to reconstruct the elastic document; note the ":" is
        # automatically converted to a "." in the filename.
        elastic_relax_maker.write_additional_data["transformations:json"] = ts

        # create the job
        relax_job = elastic_relax_maker.make(
            deformed_structure, prev_vasp_dir=prev_vasp_dir
        )
        relax_job.append_name(f" {i + 1}/{len(deformations)}")
        relaxations.append(relax_job)

        # extract the outputs we want
        output = {
            "stress": relax_job.output.output.stress,
            "deformation": deformation,
            "uuid": relax_job.output.uuid,
            "job_dir": relax_job.output.dir_name,
        }

        outputs.append(output)

    relax_flow = Flow(relaxations, outputs)
    return Response(replace=relax_flow)


@job(output_schema=ElasticDocument)
def fit_elastic_tensor(
    structure: Structure,
    deformation_data: list[dict],
    equilibrium_stress: Matrix3D | None = None,
    order: int = 2,
    fitting_method: str = SETTINGS.ELASTIC_FITTING_METHOD,
    symprec: float = SETTINGS.SYMPREC,
):
    """
    Analyze stress/strain data to fit the elastic tensor and related properties.

    Parameters
    ----------
    structure : ~pymatgen.core.structure.Structure
        A pymatgen structure.
    deformation_data : list of dict
        The deformation data, as a list of dictionaries, each containing the keys
        "stress", "deformation".
    equilibrium_stress : None or tuple of tuple of float
        The equilibrium stress of the (relaxed) structure, if known.
    order : int
        Order of the tensor expansion to be fitted. Can be either 2 or 3.
    fitting_method : str
        The method used to fit the elastic tensor. See pymatgen for more details on the
        methods themselves. The options are:

        - "finite_difference" (note this is required if fitting a 3rd order tensor)
        - "independent"
        - "pseudoinverse"
    symprec : float
        Symmetry precision for deriving symmetry equivalent deformations. If
        ``symprec=None``, then no symmetry operations will be applied.
    """
    stresses = []
    deformations = []
    uuids = []
    job_dirs = []
    for data in deformation_data:

        # stress could be none if the deformation calculation failed
        if data["stress"] is None:
            continue

        stresses.append(Stress(data["stress"]))
        deformations.append(Deformation(data["deformation"]))
        uuids.append(data["uuid"])
        job_dirs.append(data["job_dir"])

    logger.info("Analyzing stress/strain data")

    elastic_doc = ElasticDocument.from_stresses(
        structure,
        stresses,
        deformations,
        uuids,
        job_dirs,
        fitting_method=fitting_method,
        order=order,
        equilibrium_stress=equilibrium_stress,
        symprec=symprec,
    )
    return elastic_doc
