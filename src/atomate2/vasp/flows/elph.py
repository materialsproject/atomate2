"""Flows for electron phonon calculations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker, OnMissing

from atomate2.vasp.flows.core import (
    DoubleRelaxMaker,
    HSEUniformBandStructureMaker,
    UniformBandStructureMaker,
)
from atomate2.vasp.jobs.core import (
    HSEBSMaker,
    HSEStaticMaker,
    NonSCFMaker,
    StaticMaker,
    TightRelaxMaker,
)
from atomate2.vasp.jobs.elph import (
    DEFAULT_ELPH_TEMPERATURES,
    DEFAULT_MIN_SUPERCELL_LENGTH,
    SupercellElectronPhononDisplacedStructureMaker,
    calculate_electron_phonon_renormalisation,
    run_elph_displacements,
)
from atomate2.vasp.sets.core import (
    HSEBSSetGenerator,
    HSEStaticSetGenerator,
    NonSCFSetGenerator,
    StaticSetGenerator,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class ElectronPhononMaker(Maker):
    """
    Maker to create electron phonon displaced structures and band gap renormalisation.

    This workflow contains:

    1. An initial tight structure relaxation (optional if relax_maker set to None).
    2. A static calculation to determine if the material is magnetic.
    3. A finite-difference calculation to generate the electron-phonon displaced
       structures. This is performed after a supercell transformation is applied. The
       goal is to find a cubicish supercell with lengths > 15 Å. The size of the
       supercell can be modified using the ``min_supercell_length`` option.
    4. A uniform band structure calculation on each of the displaced structures
       (comprising a static calculation and uniform non-self-consistent field
       calculation).
    5. A uniform band structure calculation on the bulk undisplaced supercell
       structure, this is used as the ground state for calculating the band gap
       renormalisation.

    .. warning::
        It is not recommended to disable the tight relaxation unless you know what you
        are doing. Accurate forces are required to obtained non-imaginary phonon
        frequencies.

    .. warning::
        Currently no check is performed to ensure all phonon frequencies are real.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperatures : tuple of float
        Temperatures at which electron-phonon interactions are calculated.
    min_supercell_length : float
        Minimum supercell length in A. See :obj:`.CubicSupercellTransformation` for more
        details.
    relax_maker : BaseVaspMaker
        Maker to use for the initial structure relaxation.
    static_maker : BaseVaspMaker
        Maker to use for the static calculation on the relaxed structure.
    elph_displacement_maker : SupercellElectronPhononDisplacedStructureMaker
        Maker to use to generate the supercell and calculate electron phonon displaced
        structures.
    uniform_maker : BaseVaspMaker
        Maker to use to run the density of states on the displaced structures and
        bulk supercell structure.
    """

    name: str = "electron phonon"
    temperatures: tuple[float, ...] = DEFAULT_ELPH_TEMPERATURES
    min_supercell_length: float = DEFAULT_MIN_SUPERCELL_LENGTH
    relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    static_maker: BaseVaspMaker = field(default_factory=StaticMaker)
    elph_displacement_maker: SupercellElectronPhononDisplacedStructureMaker = field(
        default_factory=SupercellElectronPhononDisplacedStructureMaker
    )
    uniform_maker: BaseVaspMaker = field(
        default_factory=lambda: UniformBandStructureMaker(
            static_maker=StaticMaker(
                input_set_generator=StaticSetGenerator(
                    auto_ispin=True,
                    user_incar_settings={"KSPACING": None, "EDIFF": 1e-5},
                    user_kpoints_settings={"reciprocal_density": 50},
                ),
            ),
            bs_maker=NonSCFMaker(
                input_set_generator=NonSCFSetGenerator(
                    reciprocal_density=100,  # dense BS mesh
                    user_incar_settings={"LORBIT": 10},  # disable site projections
                ),
                task_document_kwargs={
                    "strip_bandstructure_projections": True,
                    "strip_dos_projections": True,
                },
            ),
        )
    )

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """Create a electron-phonon coupling workflow.

        Parameters
        ----------
        structure: .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            An electron phonon coupling workflow.
        """
        jobs = []

        if self.relax_maker is not None:
            # optionally relax the structure
            relax = self.relax_maker.make(structure, prev_dir=prev_dir)
            jobs.append(relax)
            structure = relax.output.structure
            prev_dir = relax.output.dir_name

        static = self.static_maker.make(structure, prev_dir=prev_dir)

        # update temperatures and supercell size for elph maker but make sure to not
        # overwrite original maker
        elph_maker = deepcopy(self.elph_displacement_maker)
        elph_maker.temperatures = self.temperatures
        elph_maker.min_supercell_length = self.min_supercell_length
        elph = elph_maker.make(static.output.structure, prev_dir=static.output.dir_name)

        # use static as prev_dir so we don't inherit elph settings; using a prev
        # directory is useful as we can turn off magnetism if necessary which gives a
        # reasonable speedup
        supercell_dos = self.uniform_maker.make(
            elph.output.structure, prev_dir=static.output.dir_name
        )
        supercell_dos.append_name(" bulk supercell")

        displaced_doses = run_elph_displacements(
            elph.output.calcs_reversed[0].output.elph_displaced_structures.temperatures,
            elph.output.calcs_reversed[0].output.elph_displaced_structures.structures,
            self.uniform_maker,
            prev_dir=static.output.dir_name,
            original_structure=static.output.structure,
            supercell_structure=elph.output.structure,
        )

        renorm = calculate_electron_phonon_renormalisation(
            displaced_doses.output["temperatures"],
            displaced_doses.output["band_structures"],
            displaced_doses.output["structures"],
            displaced_doses.output["uuids"],
            displaced_doses.output["dirs"],
            supercell_dos.output.vasp_objects["bandstructure"],
            supercell_dos.output.structure,
            supercell_dos.output.uuid,
            supercell_dos.output.dir_name,
            elph.output.uuid,
            elph.output.dir_name,
            static.output.structure,
        )

        # allow some of the displacements to fail
        renorm.config.on_missing_references = OnMissing.NONE

        jobs.extend([static, elph, supercell_dos, displaced_doses, renorm])
        return Flow(jobs, renorm.output, name=self.name)


@dataclass
class HSEElectronPhononMaker(ElectronPhononMaker):
    """
    Maker to create electron phonon displaced structures and HSE gap renormalisation.

    This workflow contains:

    1. An initial PBEsol tight structure relaxation (optional if relax_maker set to
       None).
    2. A PBEsol static calculation to determine if the material is magnetic.
    3. A PBEsol finite-difference calculation to generate the electron-phonon displaced
       structures. This is performed after a supercell transformation is applied. The
       goal is to find a cubicish supercell with lengths > 15 Å. The size of the
       supercell can be modified using the ``min_supercell_length`` option.
    4. A HSE06 uniform band structure calculation on each of the displaced structures
       (comprising a static calculation and uniform non-self-consistent field
       calculation).
    5. A HSE06 uniform band structure calculation on the bulk undisplaced supercell
       structure, this is used as the ground state for calculating the band gap
       renormalisation.

    .. note::
        The only difference between this workflow and :obj:`ElectronPhononMaker` is that
        the uniform electronic structures are obtained using HSE06 rather than PBEsol.
        All other calculations (relaxations, phonon frequencies etc, are still obtained
        using PBEsol).

    .. warning::
        It is not recommended to disable the tight relaxation unless you know what you
        are doing. Accurate forces are required to obtained non-imaginary phonon
        frequencies.

    .. warning::
        Currently no check is performed to ensure all phonon frequencies are real.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    temperatures : tuple of float
        Temperatures at which electron-phonon interactions are calculated.
    min_supercell_length : float
        Minimum supercell length in A. See :obj:`.CubicSupercellTransformation` for more
        details.
    relax_maker : BaseVaspMaker
        Maker to use for the initial structure relaxation.
    static_maker : BaseVaspMaker
        Maker to use for the static calculation on the relaxed structure.
    elph_displacement_maker : SupercellElectronPhononDisplacedStructureMaker
        Maker to use to generate the supercell and calculate electron phonon displaced
        structures.
    uniform_maker : BaseVaspMaker
        Maker to use to run the density of states on the displaced structures and
        bulk supercell structure.
    """

    name: str = "hse electron phonon"
    uniform_maker: BaseVaspMaker = field(
        default_factory=lambda: HSEUniformBandStructureMaker(
            static_maker=HSEStaticMaker(
                input_set_generator=HSEStaticSetGenerator(
                    auto_ispin=True,
                    user_incar_settings={"KSPACING": None, "EDIFF": 1e-5},
                    user_kpoints_settings={"reciprocal_density": 64},
                )
            ),
            bs_maker=HSEBSMaker(
                input_set_generator=HSEBSSetGenerator(
                    user_incar_settings={"LORBIT": 10},  # disable site projections
                    user_kpoints_settings={"reciprocal_density": 200},  # dense BS mesh
                ),
                task_document_kwargs={
                    "strip_bandstructure_projections": True,
                    "strip_dos_projections": True,
                },
            ),
        )
    )
