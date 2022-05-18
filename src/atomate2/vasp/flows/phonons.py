"""Flows for calculating elastic constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2 import SETTINGS
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.jobs.core import TightRelaxMaker
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker, generate_phonon_displacements, \
    run_phonon_displacements, generate_frequencies_eigenvectors
from atomate2.vasp.sets.core import StaticSetGenerator

__all__ = ["PhononMaker"]


@dataclass
class PhononMaker(Maker):
    """
    Maker to calculate harmonic phonons with VASP and Phonopy.

    Calculate the harmonic phonons of a material. Initially, a tight structural
    relaxation is performed to obtain a structure without forces on the atoms.
    Subsequently, supercells with one displaced atom are generated and accurate
    forces are computed for these structures. With the help of phonopy, these
    forces are then converted into a dynamical matrix. To correct for polarization effects,
    a correction of the dynamical matrix based on BORN charges can be performed.
    Finally, phonon densities of states, phonon band structures and thermodynamic properties are computed.

    .. Note::
        It is heavily recommended to symmetrize the structure before passing it to
        this flow. Otherwise, a different space group might be detected and too many displacement calculations
        will be generated.
        It is recommended to check the convergence parameters here and adjust them if necessary. The default might
        not be strict enough for your specific case.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    sym_reduce : bool
        Whether to reduce the number of deformations using symmetry.
    symprec : float
        Symmetry precision to use in the reduction of symmetry.
    displacement: float
        displacement distance for phonons
    min_length: float
        min length of the supercell that will be build
    conventional: bool
        if true, the supercell will be built from the conventional cell and all properties will be related
         to the conventional cell
    bulk_relax_maker : .BaseVaspMaker or None
        A maker to perform a tight relaxation on the bulk. Set to ``None`` to skip the
        bulk relaxation
    generate_phonon_diplsacements_kwargs: dict
        keyword arguments paseed to :oj: enerate_phonon_displacements
    run_phonon_displacements_kwargs : dict
        Keyword arguments passed to :obj:`run_phonon_displacements__kwargs`.
    born_maker: .BaseVaspMaker or None
        Maker to compute the BORN charges.
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.
    generate_frequencies_eigenvectors_kwargs : dict
        Keyword arguments passed to :obj:`generate_frequencies_eigenvectors`.
    """

    name: str = "phonon"
    sym_reduce: bool = True
    symprec: float = SETTINGS.SYMPREC
    displacement: float = 0.01
    min_length: float = 20.0
    conventional: bool = False
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    generate_phonon_displacements_kwargs: dict = field(default_factory=dict)
    run_phonon_displacements_kwargs: dict = field(default_factory=dict)
    born_maker: BaseVaspMaker = field(default_factory=StaticSetGenerator)
    phonon_displacement_maker: BaseVaspMaker = field(default_factory=PhononDisplacementMaker)
    generate_frequencies_eigenvectors_kwargs: dict = field(default_factory=dict)

    def make(
            self,
            structure: Structure,
            prev_vasp_dir: str | Path | None = None,
    ):
        """
        Make flow to calculate the elastic constant.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure.
        prev_vasp_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.
        """
        jobs = []

        # convert to primitive cell
        sga = SpacegroupAnalyzer(structure, symprec=self.symprec)
        structure = sga.get_primitive_standard_structure()

        if self.bulk_relax_maker is not None:
            # optionally relax the structure
            bulk = self.bulk_relax_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
            jobs.append(bulk)
            structure = bulk.output.structure
        if self.conventional:
            sga = SpacegroupAnalyzer(structure, symprec=self.symprec)
            structure = sga.get_conventional_standard_structure()
        # generate the displacements
        displacements = generate_phonon_displacements(structure=structure, symprec=self.symprec,
                                                      sym_reduce=self.sym_reduce, displacement=self.displacement,
                                                      min_length=self.min_length,
                                                      conventional=self.conventional,
                                                      **self.generate_phonon_displacements_kwargs)
        jobs.append(displacements)

        # perform the phonon displacement calculations
        vasp_displacement_calcs = run_phonon_displacements(displacements.output,
                                                           phonon_maker=self.phonon_displacement_maker,
                                                           **self.run_phonon_displacements_kwargs)
        jobs.append(vasp_displacement_calcs)

        # Computation of BORN charges
        if self.born_maker is None:
            self.born_maker = StaticSetGenerator(lepsilon=True)
        if not self.born_maker.lepsilon:
            raise ValueError("born_maker must include lepsilon=True")
        born_job = StaticMaker(input_set_generator=self.born_maker).make(structure=structure)
        jobs.append(born_job)

        # Currently we access forces via filepathes to avoid large data transfer

        phonon_collect = generate_frequencies_eigenvectors(structure=structure,
                                                           displacement_data=vasp_displacement_calcs.output,
                                                           symprec=self.symprec, symreduc=self.sym_reduce,
                                                           displacement=self.displacement,
                                                           min_length=self.min_length,
                                                           conventional=self.conventional,
                                                           born_data=born_job.output.dir_name,
                                                           **self.generate_frequencies_eigenvectors_kwargs)
        jobs.append(phonon_collect)
        # # create a flow including all jobs for a phonon computation
        my_flow = Flow(jobs, phonon_collect.output)
        return my_flow
