"""Flows for calculating elastic constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from atomate2 import SETTINGS
from atomate2.common.schemas.math import Matrix3D
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import DielectricMaker, StaticMaker, TightRelaxMaker
from atomate2.vasp.jobs.phonons import (
    PhononDisplacementMaker,
    generate_frequencies_eigenvectors,
    generate_phonon_displacements,
    get_supercell_size,
    run_phonon_displacements,
    structure_to_conventional,
    structure_to_primitive,
)

__all__ = ["PhononMaker"]


@dataclass
class PhononMaker(Maker):
    """
    Maker to calculate harmonic phonons with VASP and Phonopy.

    Calculate the harmonic phonons of a material. Initially, a tight structural
    relaxation is performed to obtain a structure without forces on the atoms.
    Subsequently, supercells with one displaced atom are generated and accurate
    forces are computed for these structures. With the help of phonopy, these
    forces are then converted into a dynamical matrix. To correct for polarization
    effects, a correction of the dynamical matrix based on BORN charges can
    be performed.     Finally, phonon densities of states, phonon band structures
    and thermodynamic properties are computed.

    .. Note::
        It is heavily recommended to symmetrize the structure before passing it to
        this flow. Otherwise, a different space group might be detected and too
        many displacement calculations will be generated.
        It is recommended to check the convergence parameters here and
        adjust them if necessary. The default might not be strict enough
        for your specific case.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    sym_reduce : bool
        Whether to reduce the number of deformations using symmetry.
    symprec : float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
    symprec_phonopy : float
        Symmetry precision for all phonopy-related symmetry searches
    displacement: float
        displacement distance for phonons
    min_length: float
        min length of the supercell that will be build
    supercell_matrix: list
        instead of min_length, also a supercell_matrix can
         be given, e.g. [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]
    use_primitive_standard_structure: bool
        this will enforce to start the phonon computation
        from the primitive standard structure
        according to Setyawan, W., & Curtarolo, S. (2010).
        High-throughput electronic band structure calculations:
        Challenges and tools. Computational Materials Science,
        49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
        This makes it possible to use certain k-path definitions
        with this workflow. Otherwise, we must rely on seekpath
    use_conventional_standard_structure: bool
        this will enforce to start the phonon computation
        from the conventional standard structure
        according to Setyawan, W., & Curtarolo, S. (2010).
        High-throughput electronic band structure calculations:
        Challenges and tools. Computational Materials Science,
        49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
         We will however use seekpath and primitive structures
        from phonopy to compute the phonon band structure
    bulk_relax_maker : .BaseVaspMaker or None
        A maker to perform a tight relaxation on the bulk.
        Set to ``None`` to skip the
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
    kpath_scheme: str
        scheme to generate kpoints. Please be aware t
        hat you can only use seekpath with any kind of cell
    code: str
        determines the dft code. currently only vasp is implemented
    """

    # TODO: add some unit conversion factors
    #  to easily use other codes to compute phonons?
    name: str = "phonon"
    sym_reduce: bool = True
    symprec: float = SETTINGS.SYMPREC
    symprec_phonopy: float = 0.001
    displacement: float = 0.01
    min_length: float | None = 20.0
    use_primitive_standard_structure: bool = False
    use_conventional_standard_structure: bool = False
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    static_energy_maker: BaseVaspMaker | None = field(default_factory=StaticMaker)
    generate_phonon_displacements_kwargs: dict = field(default_factory=dict)
    run_phonon_displacements_kwargs: dict = field(default_factory=dict)
    born_maker: BaseVaspMaker | None = field(default_factory=DielectricMaker)
    phonon_displacement_maker: BaseVaspMaker = field(
        default_factory=PhononDisplacementMaker
    )
    create_thermal_displacements: bool = True
    generate_frequencies_eigenvectors_kwargs: dict = field(default_factory=dict)
    kpath_scheme: str = "seekpath"
    code: str = "vasp"

    def make(
        self,
        structure: Structure,
        prev_vasp_dir: str | Path | None = None,
        born_manual: List[Matrix3D] | None = None,
        full_born: bool = True,
        epsilon_static_manual: Matrix3D | None = None,
        supercell_matrix: Matrix3D | None = None,
        total_dft_energy: float | None = None,
    ):
        """
        Make flow to calculate the elastic constant.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure.
        prev_vasp_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.
        born_manual: Matrix3D
            Instead of recomputing born charges and epsilon,
            these values can also be provided manually. born_maker has to
            be None
            if full_born==False, it will expect Phonopy convention for symmetrically
            inequivalent atoms
            if fullborn==True, it will expect VASP convention with information for
            every atom in unit cell. Please be careful when converting
            structures within in this workflow
        full_born: bool
            reduced born file (only symmerically inequivalent atoms)
            or full information on born
        epsilon_static_manual: Matrix3D
            Instead of recomputing born charges and epsilon,
            these values can also be provided manually. born_maker has to
            be None

        """
        # TODO: check if there is a better way to access
        #  born charges than via outcar["born"]?
        # TODO: make sure all parameters are tight enough
        #  for phonons! Cross-check with A.B. workflow
        # TODO: can we add some kind of convergence test?
        # TODO: potentially improve supercell transformation -
        #  does not always find cell with lattice parameters close to
        # 90
        # TODO: make sure the workflow works if BORN=0

        if (
            not self.use_primitive_standard_structure
            and self.kpath_scheme != "seekpath"
        ):
            raise ValueError(
                "You can only use other kpath schemes with the primitive standard structure"
            )

        if self.kpath_scheme not in [
            "seekpath",
            "hinuma",
            "setyawan_curtarolo",
            "latimer_munro",
            "all_pymatgen",
        ]:
            raise ValueError("kpath scheme is not implemented")

        jobs = []

        if self.use_primitive_standard_structure:
            # These structures are compatible with many
            # of the kpath algorithms that are used for Materials Project
            prim_job = structure_to_primitive(structure, self.symprec)
            jobs.append(prim_job)
            structure = prim_job.output
        elif self.use_conventional_standard_structure:
            # it could be beneficial to use conventional
            # standard structures to arrive faster at supercells with right
            # angels
            conv_job = structure_to_conventional(structure, self.symprec)
            jobs.append(conv_job)
            structure = conv_job.output

        # if supercell_matrix is None, supercell size will be determined
        if supercell_matrix is not None:
            self.supercell_matrix = None
        else:
            supercell_job = get_supercell_size(structure, self.min_length)
            jobs.append(supercell_job)
            self.supercell_matrix = supercell_job.output

        if self.bulk_relax_maker is not None:
            # optionally relax the structure
            bulk = self.bulk_relax_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
            jobs.append(bulk)
            structure = bulk.output.structure

        # get a phonon object from phonopy
        displacements = generate_phonon_displacements(
            structure=structure,
            supercell_matrix=self.supercell_matrix,
            displacement=self.displacement,
            sym_reduce=self.sym_reduce,
            symprec=self.symprec_phonopy,
            use_standard_primitive=self.use_primitive_standard_structure,
            kpath_scheme=self.kpath_scheme,
            code=self.code,
        )
        jobs.append(displacements)

        # perform the phonon displacement calculations
        vasp_displacement_calcs = run_phonon_displacements(
            displacements=displacements.output,
            structure=structure,
            supercell_matrix=self.supercell_matrix,
            phonon_maker=self.phonon_displacement_maker,
        )
        jobs.append(vasp_displacement_calcs)

        # Computation of BORN charges
        if self.born_maker is not None and (
            born_manual is None or epsilon_static_manual is None
        ):
            born_job = self.born_maker.make(structure)
            jobs.append(born_job)

        # Computation of static energy
        if self.static_energy_maker is not None:
            static_job = self.static_energy_maker.make(structure=structure)
            jobs.append(static_job)
            total_energy = static_job.output.output.energy
            static_run_job_dir = static_job.output.dir_name
            static_run_uuid = static_job.output.uuid
        else:
            total_energy = total_dft_energy
            static_run_job_dir = None
            static_run_uuid = None

        if self.born_maker is not None:
            epsilon_static = born_job.output.calcs_reversed[0].output.epsilon_static
            born = born_job.output.calcs_reversed[0].output.outcar["born"]
            full_born = True
            born_run_job_dir = born_job.output.dir_name
            born_run_uuid = born_job.output.uuid
        else:
            epsilon_static = epsilon_static_manual
            born = born_manual
            born_run_job_dir = None
            born_run_uuid = None

        phonon_collect = generate_frequencies_eigenvectors(
            supercell_matrix=self.supercell_matrix,
            displacement=self.displacement,
            sym_reduce=self.sym_reduce,
            symprec=self.symprec_phonopy,
            use_standard_primitive=self.use_primitive_standard_structure,
            kpath_scheme=self.kpath_scheme,
            code=self.code,
            structure=structure,
            displacement_data=vasp_displacement_calcs.output,
            epsilon_static=epsilon_static,
            born=born,
            full_born=full_born,
            total_energy=total_energy,
            static_run_job_dir=static_run_job_dir,
            static_run_uuid=static_run_uuid,
            born_run_job_dir=born_run_job_dir,
            born_run_uuid=born_run_uuid,
            create_thermal_displacements=self.create_thermal_displacements,
            **self.generate_frequencies_eigenvectors_kwargs,
        )

        jobs.append(phonon_collect)
        # # create a flow including all jobs for a phonon computation
        my_flow = Flow(jobs, phonon_collect.output)
        return my_flow
