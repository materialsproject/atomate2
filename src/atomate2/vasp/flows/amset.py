"""Flows for calculating transport properties using VASP."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Maker, job

from atomate2 import SETTINGS
from atomate2.amset.jobs import AmsetMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.flows.elastic import ElasticMaker
from atomate2.vasp.jobs.amset import (
    DenseUniformMaker,
    HSEDenseUniformMaker,
    HSEStaticDeformationMaker,
    StaticDeformationMaker,
    calculate_deformation_potentials,
    calculate_polar_phonon_frequency,
    generate_wavefunction_coefficients,
    run_amset_deformations,
)
from atomate2.vasp.jobs.core import (
    DielectricMaker,
    HSEBSMaker,
    HSEStaticMaker,
    HSETightRelaxMaker,
    StaticMaker,
    TightRelaxMaker,
)
from atomate2.vasp.sets.core import HSEBSSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker


_DEFAULT_DOPING = (
    1e16,
    1e17,
    1e18,
    1e19,
    1e20,
    1e21,
    -1e16,
    -1e17,
    -1e18,
    -1e19,
    -1e20,
    -1e21,
)
_DEFAULT_TEMPERATURES = (200, 300, 400, 500, 600, 700, 800, 900, 1000)


@dataclass
class DeformationPotentialMaker(Maker):
    """
    Maker to generate acoustic deformation potentials for amset.

    .. Note::
        It is heavily recommended to symmetrize the structure before passing it to this
        flow. Otherwise, the deformation potentials may not be aligned correctly.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    symprec : float
        Symmetry precision to use in the reduction of symmetry.
    elastic_relax_maker : .BaseVaspMaker
        Maker used to generate elastic relaxations.
    """

    name: str = "deformation potential"
    symprec: float = SETTINGS.SYMPREC
    static_deformation_maker: BaseVaspMaker = field(
        default_factory=StaticDeformationMaker
    )

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        ibands: tuple[list[int], list[int]] = None,
    ) -> Flow:
        """Make flow to calculate acoustic deformation potentials.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure.
        prev_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.
        ibands : tuple of list of int
            Which bands to include in the deformation.h5 file. Given as a tuple of one
            or two lists (one for each spin channel). The bands indices are zero
            indexed.
        """
        bulk = self.static_deformation_maker.make(structure, prev_dir=prev_dir)
        bulk.append_name("bulk ", prepend=True)

        # all deformation calculations need to be on the same k-point mesh, to achieve
        # this we override user_kpoints_settings with the desired k-points
        bulk_kpoints = bulk.output.output.orig_inputs.kpoints
        deformation_maker = deepcopy(self.static_deformation_maker)
        deformation_maker.input_set_generator.user_kpoints_settings = bulk_kpoints

        # generate and run the deformations
        vasp_deformation_calcs = run_amset_deformations(
            bulk.output.structure,
            symprec=self.symprec,
            prev_dir=bulk.output.dir_name,
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
            output=deformation_potentials.output,
            name=self.name,
        )


@dataclass
class VaspAmsetMaker(Maker):
    """
    Maker to calculate transport properties using AMSET with VASP calculation as input.

    .. Note::
        It is heavily recommended to symmetrize the structure before passing it to
        this flow. Otherwise, the transport properties may not lie along the
        correct axes.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    doping : tuple of float
        Doping concentrations at which to calculate transport properties.
    temperatures : tuple of float
        Temperatures at which to calculate transport properties.
    use_hse_gap : bool
        Whether to perform an HSE06 calculation to calculate the band gap for use in
        AMSET. This can impact the results for small band gap materials.
    relax_maker : .BaseVaspMaker
        A maker to perform a tight relaxation on the bulk. Set to ``None`` to skip the
        bulk relaxation.
    static_maker : .BaseVaspMaker
        The maker to use for the initial static calculation.
    dense_uniform_maker : .BaseVaspMaker
        The maker to use for dense uniform calculations.
    dielectric_maker : .BaseVaspMaker
        The maker to use for calculating dielectric constants.
    elastic_maker : .ElasticMaker
        The maker to use for calculating elastic constants.
    deformation_potential_maker : .DeformationPotentialMaker
        The maker to use for calculating acoustic deformation potentials.
    hse_gap_maker : .BaseVaspMaker
        The maker to use for calculating the band gap using HSE06. Note, this maker is
        only used if ``use_hse_gap=True``.
    amset_maker : .AmsetMaker
        The maker to use for running AMSET calculations.
    """

    name: str = "VASP amset"
    doping: tuple[float, ...] = _DEFAULT_DOPING
    temperatures: tuple[float, ...] = _DEFAULT_TEMPERATURES
    use_hse_gap: bool = True
    amset_settings: dict = field(default_factory=dict)
    relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    static_maker: BaseVaspMaker = field(default_factory=StaticMaker)
    dense_uniform_maker: BaseVaspMaker = field(default_factory=DenseUniformMaker)
    dielectric_maker: BaseVaspMaker = field(default_factory=DielectricMaker)
    elastic_maker: ElasticMaker = field(
        default_factory=lambda: ElasticMaker(bulk_relax_maker=None)
    )
    deformation_potential_maker: DeformationPotentialMaker = field(
        default_factory=DeformationPotentialMaker
    )
    hse_gap_maker: BaseVaspMaker = field(
        default_factory=lambda: HSEBSMaker(
            input_set_generator=HSEBSSetGenerator(user_incar_settings={"EDIFF": 1e-5})
        )
    )
    amset_maker: AmsetMaker = field(default_factory=lambda: AmsetMaker(resubmit=True))

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """Make flow to calculate electronic transport properties using AMSET and VASP.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure.
        prev_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.
        """
        jobs = []

        if self.relax_maker is not None:
            # optionally relax the structure
            bulk = self.relax_maker.make(structure, prev_dir=prev_dir)
            jobs.append(bulk)
            structure = bulk.output.structure
            prev_dir = bulk.output.dir_name

        static = self.static_maker.make(structure, prev_dir=prev_dir)

        # dense band structure for eigenvalues and wave functions
        dense_bs = self.dense_uniform_maker.make(
            static.output.structure, prev_dir=static.output.dir_name
        )

        # elastic constant
        elastic = self.elastic_maker.make(
            static.output.structure,
            prev_dir=static.output.dir_name,
            equilibrium_stress=static.output.output.stress,
        )

        # dielectric constant
        dielectric = self.dielectric_maker.make(
            static.output.structure, prev_dir=static.output.dir_name
        )

        # polar phonon frequency
        phonon_frequency = calculate_polar_phonon_frequency(
            dielectric.output.structure,
            dielectric.output.calcs_reversed[0].output.normalmode_frequencies,
            dielectric.output.calcs_reversed[0].output.normalmode_eigenvecs,
            dielectric.output.calcs_reversed[0].output.outcar["born"],
        )

        # wavefunction coefficients
        wavefunction = generate_wavefunction_coefficients(dense_bs.output.dir_name)

        # deformation potentials
        deformation = self.deformation_potential_maker.make(
            static.output.structure,
            prev_dir=static.output.dir_name,
            ibands=wavefunction.output["ibands"],
        )

        # sum high-frequency dielectric and ionic contribution to get static dielectric
        # note: the naming of dielectric constants in VASP and pymatgen is wrong
        high_freq_dielectric = dielectric.output.calcs_reversed[0].output.epsilon_static
        static_dielectric = job(np.sum)(
            [
                dielectric.output.calcs_reversed[0].output.epsilon_ionic,
                high_freq_dielectric,
            ],
            axis=0,
        )

        jobs += [
            static,
            dense_bs,
            elastic,
            dielectric,
            phonon_frequency,
            wavefunction,
            deformation,
            static_dielectric,
        ]

        # compile all property calculations and generate settings for AMSET
        # set doping and temperature but be careful not to override user selections
        settings = {
            "doping": self.doping,
            "temperatures": self.temperatures,
            "pop_frequency": phonon_frequency.output["frequency"],
            "elastic_constant": elastic.output.elastic_tensor.raw,
            "high_frequency_dielectric": high_freq_dielectric,
            "static_dielectric": static_dielectric.output,
            "deformation_potential": "deformation.h5",
            "print_log": False,
            "interpolation_factor": 5,
            "free_carrier_screening": True,
        }

        if self.use_hse_gap and "bandgap" not in self.amset_settings:
            gap = self.hse_gap_maker.make(
                dense_bs.output.structure,
                prev_dir=dense_bs.output.dir_name,
                mode="gap",
            )
            settings["bandgap"] = gap.output.output.bandgap
            jobs.append(gap)

        # apply the user settings
        settings.update(self.amset_settings)

        # amset transport properties
        amset = self.amset_maker.make(
            settings,
            wavefunction_dir=wavefunction.output["dir_name"],
            deformation_dir=deformation.output["dir_name"],
            bandstructure_dir=dense_bs.output.dir_name,
        )
        jobs.append(amset)

        return Flow(jobs, output=amset.output, name=self.name)


@dataclass
class HSEVaspAmsetMaker(Maker):
    """
    Maker to calculate transport properties using AMSET with HSE06 VASP inputs.

    .. Note::
        Dielectric and elastic constants are still calculated using PBEsol, whereas
        electronic properties, deformation potentials, and wavefunctions are calculated
        using HSE06.

    .. Note::
        It is heavily recommended to symmetrize the structure before passing it to
        this flow. Otherwise, the transport properties may not lie along the
        correct axes.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    doping : tuple of float
        Doping concentrations at which to calculate transport properties.
    temperatures : tuple of float
        Temperatures at which to calculate transport properties.
    relax_maker : .BaseVaspMaker
        A maker to perform a tight relaxation on the bulk. Set to ``None`` to skip the
        bulk relaxation.
    static_maker : .BaseVaspMaker
        The maker to use for the initial static calculation.
    dense_uniform_maker : .BaseVaspMaker
        The maker to use for dense uniform calculations.
    dielectric_maker : .BaseVaspMaker
        The maker to use for calculating dielectric constants.
    elastic_maker : .ElasticMaker
        The maker to use for calculating elastic constants.
    deformation_potential_maker : .DeformationPotentialMaker
        The maker to use for calculating acoustic deformation potentials.
    amset_maker : .AmsetMaker
        The maker to use for running AMSET calculations.
    """

    name: str = "hse VASP amset"
    doping: tuple[float, ...] = _DEFAULT_DOPING
    temperatures: tuple[float, ...] = _DEFAULT_TEMPERATURES
    amset_settings: dict = field(default_factory=dict)
    relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(HSETightRelaxMaker())
    )
    static_maker: BaseVaspMaker = field(default_factory=HSEStaticMaker)
    dense_uniform_maker: BaseVaspMaker = field(default_factory=HSEDenseUniformMaker)
    deformation_potential_maker: DeformationPotentialMaker = field(
        default_factory=lambda: DeformationPotentialMaker(
            static_deformation_maker=HSEStaticDeformationMaker()
        )
    )
    dielectric_maker: BaseVaspMaker = field(default_factory=DielectricMaker)
    elastic_maker: ElasticMaker = field(
        default_factory=lambda: ElasticMaker(bulk_relax_maker=None)
    )
    amset_maker: AmsetMaker = field(default_factory=lambda: AmsetMaker(resubmit=True))

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """Make flow to calculate electronic transport properties using AMSET and VASP.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure.
        prev_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.
        """
        jobs = []

        if self.relax_maker is not None:
            # optionally relax the structure
            bulk = self.relax_maker.make(structure, prev_dir=prev_dir)
            jobs.append(bulk)
            structure = bulk.output.structure
            prev_dir = bulk.output.dir_name

        static = self.static_maker.make(structure, prev_dir=prev_dir)

        # dense band structure for eigenvalues and wave functions
        dense_bs = self.dense_uniform_maker.make(
            static.output.structure, prev_dir=static.output.dir_name
        )

        # elastic constant
        elastic = self.elastic_maker.make(
            static.output.structure,
            prev_dir=static.output.dir_name,
            equilibrium_stress=static.output.output.stress,
        )

        # dielectric constant
        dielectric = self.dielectric_maker.make(
            static.output.structure, prev_dir=static.output.dir_name
        )

        # polar phonon frequency
        phonon_frequency = calculate_polar_phonon_frequency(
            dielectric.output.structure,
            dielectric.output.calcs_reversed[0].output.normalmode_frequencies,
            dielectric.output.calcs_reversed[0].output.normalmode_eigenvecs,
            dielectric.output.calcs_reversed[0].output.outcar["born"],
        )

        # wavefunction coefficients
        wavefunction = generate_wavefunction_coefficients(dense_bs.output.dir_name)

        # deformation potentials
        deformation = self.deformation_potential_maker.make(
            static.output.structure,
            prev_dir=static.output.dir_name,
            ibands=wavefunction.output["ibands"],
        )

        # sum high-frequency dielectric and ionic contribution to get static dielectric
        # note: the naming of dielectric constants in VASP and pymatgen is wrong
        high_freq_dielectric = dielectric.output.calcs_reversed[0].output.epsilon_static
        static_dielectric = job(np.sum)(
            [
                dielectric.output.calcs_reversed[0].output.epsilon_ionic,
                high_freq_dielectric,
            ],
            axis=0,
        )

        jobs += [
            static,
            dense_bs,
            elastic,
            dielectric,
            phonon_frequency,
            wavefunction,
            deformation,
            static_dielectric,
        ]

        # compile all property calculations and generate settings for AMSET
        # set doping and temperature but be careful not to override user selections
        settings = {
            "doping": self.doping,
            "temperatures": self.temperatures,
            "pop_frequency": phonon_frequency.output["frequency"],
            "elastic_constant": elastic.output.elastic_tensor.raw,
            "high_frequency_dielectric": high_freq_dielectric,
            "static_dielectric": static_dielectric.output,
            "deformation_potential": "deformation.h5",
            "print_log": False,
            "interpolation_factor": 5,
            "free_carrier_screening": True,
        }

        # apply the user settings
        settings.update(self.amset_settings)

        # amset transport properties
        amset = self.amset_maker.make(
            settings,
            wavefunction_dir=wavefunction.output["dir_name"],
            deformation_dir=deformation.output["dir_name"],
            bandstructure_dir=dense_bs.output.dir_name,
        )
        jobs.append(amset)

        return Flow(jobs, output=amset.output, name=self.name)
