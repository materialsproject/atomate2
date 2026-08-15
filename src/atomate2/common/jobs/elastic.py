"""Jobs used in the calculation of elastic tensors."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Response, job
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.analysis.elasticity import Deformation, Strain, Stress
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

from atomate2 import SETTINGS
from atomate2.common.analysis.elastic import get_default_strain_states
from atomate2.common.schemas.elastic import ElasticDocument
from atomate2.common.utils import check_class_name
from atomate2.vasp.jobs.base import BaseVaspMaker

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

    from atomate2.forcefields.jobs import ForceFieldRelaxMaker
    from atomate2.torchsim import TorchSimOptimizeMaker


logger = logging.getLogger(__name__)


@job
def generate_elastic_deformations(
    structure: Structure,
    order: int = 2,
    strain_states: list[tuple[int, int, int, int, int, int]] | None = None,
    strain_magnitudes: list[float] | list[list[float]] | None = None,
    symprec: float = SETTINGS.SYMPREC,
    sym_reduce: bool = True,
) -> list[Deformation]:
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

    symprec : float
        Symmetry precision.
    sym_reduce : bool
        Whether to reduce the number of deformations using symmetry.

    Returns
    -------
    List[Deformation]
        A list of deformations.
    """
    if strain_states is None:
        strain_states = get_default_strain_states(order)

    if strain_magnitudes is None:
        strain_magnitudes = np.linspace(-0.01, 0.01, 5 + (order - 2) * 2)

    if np.array(strain_magnitudes).ndim == 1:
        strain_magnitudes = [strain_magnitudes] * len(strain_states)  # type: ignore[assignment]

    strains = []
    for state, magnitudes in zip(strain_states, strain_magnitudes, strict=True):
        strains.extend([Strain.from_voigt(m * np.array(state)) for m in magnitudes])

    # remove zero strains
    strains = [strain for strain in strains if (abs(strain) > 1e-10).any()]

    if np.linalg.matrix_rank([strain.voigt for strain in strains]) < 6:
        # TODO: check for sufficiency of input for nth order
        raise ValueError("strain list is insufficient to fit an elastic tensor")

    if sym_reduce:
        strain_mapping = symmetry_reduce(strains, structure, symprec=symprec)
        logger.info(
            f"Using symmetry to reduce number of strains from {len(strains)} to "
            f"{len(list(strain_mapping.keys()))}"
        )
        strains = list(strain_mapping.keys())

    return [s.get_deformation_matrix() for s in strains]


@job
def run_elastic_deformations(
    structure: Structure,
    deformations: list[Deformation],
    prev_dir: str | Path | None = None,
    prev_dir_argname: str | None = None,
    elastic_relax_maker: BaseVaspMaker
    | ForceFieldRelaxMaker
    | TorchSimOptimizeMaker = None,
    socket: bool = False,
) -> Response:
    """
    Run elastic deformations.

    Note, this job will replace itself with N relaxation calculations,
    or a single batched calculation using the socket interface to run all
    deformations simultaneously. This results in lower overhead as well as
    parallel relaxation of the deformations for TorchSim.

    Parameters
    ----------
    structure : Structure
        A pymatgen structure.
    deformations : list of Deformation
        The deformations to apply.
    prev_dir : str or Path or None
        A previous directory to use for copying outputs.
    prev_dir_argname: str or None
        Argument name for the prev_dir variable.
    elastic_relax_maker : .BaseVaspMaker, .ForceFieldRelaxMaker, or
        .TorchSimOptimizeMaker
        A VaspMaker, ForceFieldMaker, or TorchSimMaker to use to generate the elastic
        relaxation jobs.
    socket : bool
        If True, uses the socket-io interface to run all deformations in a single
        job, reducing overhead. In the specific case of TorchSim, this enables batching
        of all structure relaxations.
        Note: socket=True is not supported for BaseVaspMaker.
    """
    num_deformations = len(deformations)
    elastic_jobs = []
    outputs = []

    if socket and isinstance(elastic_relax_maker, BaseVaspMaker):
        raise ValueError("socket=True is not supported for BaseVaspMaker.")

    deformed_structures = []
    for deformation in deformations:
        dst = DeformStructureTransformation(deformation=deformation)
        ts = TransformedStructure(structure, transformations=[dst])
        deformed_structures.append(ts.final_structure)

        with contextlib.suppress(Exception):
            # Write details of the transformation to the transformations.json file
            elastic_relax_maker.write_additional_data["transformations:json"] = ts

    elastic_job_kwargs = {}
    if prev_dir is not None and prev_dir_argname is not None:
        elastic_job_kwargs[prev_dir_argname] = prev_dir

    if socket:
        batched_job = elastic_relax_maker.make(
            deformed_structures, **elastic_job_kwargs
        )
        batched_job.append_name(" batched_socket")
        elastic_jobs.append(batched_job)

        for idx, deformation in enumerate(deformations):
            if check_class_name(elastic_relax_maker, "TorchSimOptimizeMaker"):
                output = {
                    "stress": batched_job.output.output.stress[idx],
                    "deformation": deformation,
                    "uuid": batched_job.output.uuid,
                    "job_dir": batched_job.output.dir_name,
                }
            else:
                output = {
                    "stress": batched_job.output[idx].output.stress,
                    "deformation": deformation,
                    "uuid": batched_job.output[idx].uuid,
                    "job_dir": batched_job.output[idx].dir_name,
                }
            outputs.append(output)

    else:
        for idx, deformed_structure in enumerate(deformed_structures):
            relax_job = elastic_relax_maker.make(
                deformed_structure, **elastic_job_kwargs
            )
            relax_job.append_name(f" {idx + 1}/{num_deformations}")
            elastic_jobs.append(relax_job)

            output = {
                "stress": relax_job.output.output.stress,
                "deformation": deformations[idx],
                "uuid": relax_job.output.uuid,
                "job_dir": relax_job.output.dir_name,
            }
            outputs.append(output)

    elastic_flow = Flow(elastic_jobs, outputs)
    return Response(replace=elastic_flow)


@job(output_schema=ElasticDocument)
def fit_elastic_tensor(
    structure: Structure,
    deformation_data: list[dict],
    equilibrium_stress: Matrix3D | None = None,
    order: int = 2,
    fitting_method: str = SETTINGS.ELASTIC_FITTING_METHOD,
    symprec: float = SETTINGS.SYMPREC,
    allow_elastically_unstable_structs: bool = True,
    stress_sign_factor: float = 1.0,
    max_failed_deformations: float | None = None,
) -> ElasticDocument:
    r"""
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
    allow_elastically_unstable_structs : bool
        Whether to allow the ElasticDocument to still complete in the event that
        the structure is elastically unstable.
    stress_sign_factor: float
        Corrections for codes that define stress to be \partial E / \partial n_ij
    max_failed_deformations: int or float
        Maximum number of deformations allowed to fail to proceed with the fitting
        of the elastic tensor. If an int the absolute number of deformations. If
        a float between 0 an 1 the maximum fraction of deformations. If None any
        number of deformations allowed.
    """
    stresses = []
    deformations = []
    uuids = []
    job_dirs = []
    failed_uuids = []
    for data in deformation_data:
        # stress could be none if the deformation calculation failed
        if data["stress"] is None:
            failed_uuids.append(data["uuid"])
            continue

        stresses.append(Stress(stress_sign_factor * np.squeeze(data["stress"])))
        deformations.append(Deformation(data["deformation"]))
        uuids.append(data["uuid"])
        job_dirs.append(data["job_dir"])

    if max_failed_deformations is not None:
        if 0 < max_failed_deformations < 1:
            fraction_failed = len(failed_uuids) / len(deformation_data)
            if fraction_failed > max_failed_deformations:
                raise RuntimeError(
                    f"{fraction_failed} fraction of deformation calculations have "
                    f"failed, maximum fraction allowed: {max_failed_deformations}"
                )
        elif len(failed_uuids) > max_failed_deformations:
            raise RuntimeError(
                f"{len(failed_uuids)} deformation calculations have failed, maximum "
                f"allowed: {max_failed_deformations}"
            )

    logger.info("Analyzing stress/strain data")

    return ElasticDocument.from_stresses(
        structure,
        stresses,
        deformations,
        uuids,
        job_dirs,
        fitting_method=fitting_method,
        order=order,
        equilibrium_stress=equilibrium_stress,
        symprec=symprec,
        allow_elastically_unstable_structs=allow_elastically_unstable_structs,
        failed_uuids=failed_uuids,
    )
