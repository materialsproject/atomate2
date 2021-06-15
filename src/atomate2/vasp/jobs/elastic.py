"""Jobs used in the calculation of elastic tensors."""

from __future__ import annotations

import logging
import typing
from dataclasses import dataclass, field

import numpy as np
from jobflow import Flow, Maker, Response, job
from pymatgen.analysis.elasticity import Deformation, Strain, Stress
from pymatgen.core import SymmOp
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

from atomate2.common.analysis.elastic import get_default_strain_states
from atomate2.common.schemas.elastic import ElasticDocument
from atomate2.settings import settings
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker

if typing.TYPE_CHECKING:
    from pathlib import Path
    from typing import List, Optional, Tuple, Union

    from pymatgen.core.structure import Structure

    from atomate2.common.schemas.math import Matrix3D

logger = logging.getLogger(__name__)


@dataclass
class GenerateElasticDeformationsMaker(Maker):
    """
    Maker to generate elastic deformations..

    Parameters
    ----------
    name
        The name of jobs produced by this maker.
    order
        Order of the tensor expansion to be determined. Can be either 2 or 3.
    strain_states
        List of Voigt-notation strains, e.g. ``[(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0),
        etc]``.
    strains_magnitudes
        A list of strain magnitudes to multiply by for each strain state, e.g. ``[-0.01,
        -0.005, 0.005, 0.01]``. Alternatively, a list of lists can be specified, where
        each inner list corresponds to a specific strain state.
    conventional
        Whether to transform the structure into the conventional cell.
    symprec
        Symmetry precision.
    symmetry_reduce
        Whether to reduce the number of deformations using symmetry.
    """

    name: str = "generate deformations"
    order: int = 2
    strain_states: List[Tuple[int, int, int, int, int, int]] = None
    strain_magnitudes: Union[List[float], List[List[float]]] = None
    conventional: bool = False
    symprec: float = settings.SYMPREC
    symmetry_reduce: bool = True

    @job
    def make(self, structure: Structure):
        """
        Make a job to generate elastic deformations.

        Parameters
        ----------
        structure
            A pymatgen structure object.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the keys:

            - "deformations": containing a list of deformations.
            - "symmetry_ops": containing a list of symmetry operations or None if
              symmetry_reduce is False.
        """
        if self.conventional:
            sga = SpacegroupAnalyzer(structure, symprec=self.symprec)
            structure = sga.get_conventional_standard_structure()

        strain_states = self.strain_states
        strain_magnitudes = self.strain_magnitudes

        if strain_states is None:
            strain_states = get_default_strain_states(self.order)

        if self.strain_magnitudes is None:
            strain_magnitudes = np.linspace(-0.01, 0.01, 5 + (self.order - 2) * 2)

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
        if self.symmetry_reduce:
            deformation_mapping = symmetry_reduce(
                deformations, structure, symprec=self.symprec
            )
            deformations = deformation_mapping.keys()
            symmetry_operations = deformation_mapping.values()

        return {"deformations": deformations, "symmetry_ops": symmetry_operations}


@dataclass
class ElasticRelaxMaker(RelaxMaker):
    """
    Maker to perform an elastic relaxation.

    This is a tight relaxation where only the atom positions are allowed to relax.
    """

    name = "elastic relax"

    @job
    def make(self, structure: Structure, prev_vasp_dir: Union[str, Path] = None):
        """
        Make a job to perform a tight relaxation.

        Parameters
        ----------
        structure
            A pymatgen structure.
        prev_vasp_dir
            A previous vasp calculation directory to use for copying outputs.
        """
        incar_updates = {
            "IBRION": 2,
            "ISIF": 2,
            "ENCUT": 700,
            "EDIFF": 1e-7,
            "LAECHG": False,
            "EDIFFG": -0.001,
            "LREAL": False,
            "ALGO": "Normal",
        }
        kpoints_updates = {"grid_density": 7000}

        # make sure we don't override user settings
        incar_updates.update(self.input_set_kwargs.get("user_incar_settings", {}))
        kpoints_updates.update(self.input_set_kwargs.get("user_kpoints_settings", {}))

        self.input_set_kwargs["user_incar_settings"] = incar_updates
        self.input_set_kwargs["user_kpoints_settings"] = kpoints_updates

        # calling make would create a new job, instead we call the undecorated function
        super().make.original(structure, prev_vasp_dir=prev_vasp_dir)


@dataclass
class RunElasticDeformationsMaker(Maker):
    """Maker to run elastic deformations and extract the structural stress."""

    name: str = "run deformations"
    elastic_relax_maker: BaseVaspMaker = field(default_factory=ElasticRelaxMaker)

    @job
    def make(
        self,
        structure: Structure,
        deformations: List[Deformation],
        symmetry_ops: List[SymmOp] = None,
        prev_vasp_dir: Union[str, Path] = None,
    ):
        """
        Make a job to run the elastic deformations.

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
        """
        if symmetry_ops is not None and len(symmetry_ops) != len(deformations):
            raise ValueError(
                "Number of deformations and symmetry operations must be equal."
            )

        relaxations = []
        outputs = []
        for i, deformation in enumerate(deformations):
            # deform the structure
            dst = DeformStructureTransformation(deformation=deformation)
            deformed_structure = dst.apply_transformation(structure)

            # create the job
            relax_job = self.elastic_relax_maker.make(
                deformed_structure, prev_vasp_dir=prev_vasp_dir
            )
            relax_job.name += f" {i}"
            relaxations.append(relax_job)

            # extract the outputs we want
            output = {
                "strain": deformation.green_lagrange_strain.tolist(),
                "stress": relax_job.output.output.stress,
                "deformation_matrix": deformation.tolist(),
            }

            if symmetry_ops is not None:
                output["symmetry_ops"] = symmetry_ops[i]

            outputs.append(output)

        relax_flow = Flow(relaxations, outputs, name=self.name)
        return Response(replace=relax_flow)


@dataclass
class FitElasticTensorMaker(Maker):
    """
    Analyze stress/strain data to fit the elastic tensor and related properties.

    Parameters
    ----------
    order
        Order of the tensor expansion to be fitted. Can be either 2 or 3.
    fitting_method
        The method used to fit the elastic tensor. See pymatgen for more details on the
        methods themselves. The options are:
        - "finite_difference" (note this is required if fitting a 3rd order tensor)
        - "independent"
        - "pseudoinverse"
    """

    order: int = 2
    fitting_method: str = "finite_difference"

    @job(output_schema=ElasticDocument)
    def make(
        self,
        structure: Structure,
        strain_data: List[dict],
        equilibrium_stress: Optional[Matrix3D] = None,
    ):
        """
        Make a job to fit the elastic tensor.

        Parameters
        ----------
        structure
            A pymatgen structure.
        strain_data
            The strain data, as a list of dictionaries, each containing the keys
            "stress", "strain", "deformation_matrix", and (optionally) "symmetry_ops".
        equilibrium_stress
            The equilibrium stress of the (relaxed) structure, if known.
        """
        stresses, strains, deformations = [], [], []
        for data in strain_data:

            # data could be none if the deformation calculation failed
            if data is None:
                continue

            strain = Stress(data["strain"])
            stress = Stress(data["stress"])
            deformation = Stress(data["deformation"])

            stresses.append(stress)
            strains.append(strain)
            deformations.append(deformation)

            # add derived stresses and strains if symmetry operations are present
            for symmop in data.get("symmetry_ops", []):
                stresses.append(stress.transform(symmop))
                strains.append(strain.transform(symmop))
                deformations.append(deformation.transform(symmop))

        logger.info("Analyzing stress/strain data")

        elastic_doc = ElasticDocument.from_strains_and_stresses(
            structure,
            strains,
            stresses,
            deformations,
            self.fitting_method,
            self.order,
            equilibrium_stress=equilibrium_stress,
        )
        return elastic_doc
