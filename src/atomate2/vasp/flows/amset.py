"""Flows for calculating transport properties using VASP."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from jobflow import Flow, Maker, job
from pymatgen.core.structure import Structure

from atomate2.amset.jobs import AmsetMaker
from atomate2.settings import settings
from atomate2.vasp.flows.elastic import ElasticMaker
from atomate2.vasp.jobs.amset import (
    DenseUniformMaker,
    StaticDeformationMaker,
    calculate_deformation_potentials,
    calculate_polar_phonon_frequency,
    generate_wavefunction_coefficients,
    run_amset_deformations,
)
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import DielectricMaker, StaticMaker

__all__ = ["VaspAmsetMaker", "DeformationPotentialMaker"]


@dataclass
class DeformationPotentialMaker(Maker):
    """
    Maker to generate acoustic deformation potentials for amset.

    Parameters
    ----------
    name
        Name of the flows produced by this maker.
    symprec
        Symmetry precision to use in the reduction of symmetry.
    elastic_relax_maker
        Maker used to generate elastic relaxations.
    generate_elastic_deformations_kwargs
        Keyword arguments passed to :obj:`generate_elastic_deformations`.
    fit_elastic_tensor_kwargs
        Keyword arguments passed to :obj:`fit_elastic_tensor`.
    """

    name: str = "deformation potential"
    symprec: float = settings.SYMPREC
    static_deformation_maker: BaseVaspMaker = field(
        default_factory=StaticDeformationMaker
    )

    def make(
        self,
        structure: Structure,
        prev_vasp_dir: Union[str, Path] = None,
        ibands: Tuple[List[int], List[int]] = None,
    ):
        """
        Make flow to calculate acoustic deformation potentials.

        .. Note::
            It is heavily recommended to symmetrize the structure before passing it to
            this flow. Otherwise, the deformation potentials may not be aligned
            correctly.

        Parameters
        ----------
        structure
            A pymatgen structure.
        prev_vasp_dir
            A previous vasp calculation directory to use for copying outputs.
        ibands
            Which bands to include in the deformation.h5 file. Given as a tuple of one
            or two lists (one for each spin channel). The bands indices are zero
            indexed.
        """
        bulk = self.static_deformation_maker.make(
            structure, prev_vasp_dir=prev_vasp_dir
        )

        # all deformation calculations need to be on the same k-point mesh, to achieve
        # this we override user_kpoints_settings with the desired k-points
        bulk_kpoints = bulk.output.output.orig_inputs.kpoints
        deformation_maker = deepcopy(self.static_deformation_maker)
        deformation_maker.input_set_generator.user_kpoints_settings = bulk_kpoints

        # generate and run the deformations
        vasp_deformation_calcs = run_amset_deformations(
            bulk.output.structure,
            symprec=self.symprec,
            prev_vasp_dir=bulk.output.dir_name,
            static_deformation_maker=self.static_deformation_maker,
        )

        # generate the deformation.h5 file
        deformation_potentials = calculate_deformation_potentials(
            bulk.output.dir_name,
            vasp_deformation_calcs.output,
            symprec=self.symprec,
            ibands=ibands,
        )

        return Flow(
            jobs=[bulk, vasp_deformation_calcs, deformation_potentials],
            output=deformation_potentials.output["dir_name"],
            name=self.name,
        )


@dataclass
class VaspAmsetMaker(Maker):
    """
    Maker to calculate transport properties using AMSET with VASP calculation as input.

    Parameters
    ----------
    name
        Name of the flows produced by this maker.
    symprec
        Symmetry precision to use in the reduction of symmetry.
    fit_elastic_tensor_kwargs
        Keyword arguments passed to :obj:`fit_elastic_tensor`.
    """

    name: str = "VASP amset"
    doping: Tuple[float, ...] = (1e16, 1e17, 1e18, 1e19, 1e20, 1e21)
    temperatures: Tuple[float, ...] = (200, 300, 400, 500, 600, 700, 800, 900, 1000)
    amset_settings: dict = field(default_factory=dict)
    static_maker: BaseVaspMaker = field(default_factory=StaticMaker)
    dense_uniform_maker: BaseVaspMaker = field(default_factory=DenseUniformMaker)
    dielectric_maker: BaseVaspMaker = field(default_factory=DielectricMaker)
    elastic_maker: ElasticMaker = field(default_factory=ElasticMaker)
    deformation_potential_maker: DeformationPotentialMaker = field(
        default_factory=DeformationPotentialMaker
    )
    amset_maker: AmsetMaker = field(default_factory=AmsetMaker)

    def make(
        self,
        structure: Structure,
        prev_vasp_dir: Union[str, Path] = None,
    ):
        """
        Make flow to calculate electronic transport properties using AMSET and VASP.

        .. Note::
            It is heavily recommended to symmetrize the structure before passing it to
            this flow. Otherwise, the transport properties may not lie along the
            correct axes.

        Parameters
        ----------
        structure
            A pymatgen structure.
        prev_vasp_dir
            A previous vasp calculation directory to use for copying outputs.
        """
        static = self.static_maker.make(structure, prev_vasp_dir=prev_vasp_dir)

        # dense band structure for eigenvalues and wave functions
        dense_bs = self.dense_uniform_maker.make(
            static.output.structure, prev_vasp_dir=static.output.dir_name
        )

        # elastic constant
        elastic = self.elastic_maker.make(
            static.output.structure,
            prev_vasp_dir=static.output.dir_name,
            equilibrium_stress=static.output.output.stress,
        )

        # dielectric constant
        dielectric = self.dielectric_maker.make(
            static.output.structure, prev_vasp_dir=static.output.dir_name
        )

        # polar phonon frequency
        phonon_frequency = calculate_polar_phonon_frequency(
            dielectric.output.structure,
            dielectric.output.calcs_reversed[0].output.normalmode_frequencies,
            dielectric.output.calcs_reversed[0].output.normalmode_eigenvectors,
            dielectric.output.calcs_reversed[0].output.outcar.born,
        )

        # wavefunction coefficients
        wavefunction = generate_wavefunction_coefficients(dense_bs.output.dir_name)

        # deformation potentials
        deformation = self.deformation_potential_maker.make(
            static.output.structure,
            prev_vasp_dir=static.output.dir_name,
            ibands=wavefunction.output["ibands"],
        )

        # sum high-frequency dielectric and ionic contribution to get static dielectric
        # note: the naming of dielectric constants in VASP and pymatgen is wrong
        high_freq_dielectric = dielectric.output.calcs_reversed[0].output.epsilon_static
        static_dielectric = job(np.sum)(
            dielectric.output.calcs_reversed[0].output.epsilon_ionic,
            high_freq_dielectric,
        )

        # compile all property calculations and generate settings for AMSET
        # set doping and temperature but be careful not to override user selections
        settings = {
            "doping": self.doping,
            "temperature": self.temperatures,
            "phonon_frequency": phonon_frequency.output["frequency"],
            "elastic_constant": elastic.output.elastic_tensor.raw,
            "high_frequency_dielectric": high_freq_dielectric,
            "static_dielectric": static_dielectric.output,
            "deformation": "deformation.h5",
        }
        settings.update(self.amset_settings)

        # amset transport properties
        amset = self.amset_maker.make(
            settings,
            wavefunction_dir=wavefunction.output["dir_name"],
            deformation_dir=deformation.output["dir_name"],
        )

        return Flow(
            jobs=[
                static,
                dense_bs,
                elastic,
                dielectric,
                phonon_frequency,
                wavefunction,
                deformation,
                static_dielectric,
                amset,
            ],
            output=amset.output,
            name=self.name,
        )
