"""Flows for calculating elastic constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from atomate2.common.jobs import structure_to_conventional, structure_to_primitive
from atomate2.common.schemas.math import Matrix3D
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import DielectricMaker, StaticMaker, TightRelaxMaker
from atomate2.vasp.jobs.phonons import (
    PhononDisplacementMaker,
    generate_frequencies_eigenvectors,
    generate_phonon_displacements,
    get_supercell_size,
    get_total_energy_per_cell,
    run_phonon_displacements,
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
        and to handle all symmetry-related tasks in phonopy
    displacement: float
        displacement distance for phonons
    min_length: float
        min length of the supercell that will be built
    prefer_90_degrees: bool
        if set to True, supercell algorithm will first try to find a supercell
        with 3 90 degree angles
    get_supercell_size_kwargs: dict
        kwargs that will be passed to get_supercell_size to determine supercell size
    use_symmetrized_structure: str
        allowed strings: "primitive", "conventional", None

        - "primitive" will enforce to start the phonon computation
          from the primitive standard structure
          according to Setyawan, W., & Curtarolo, S. (2010).
          High-throughput electronic band structure calculations:
          Challenges and tools. Computational Materials Science,
          49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
          This makes it possible to use certain k-path definitions
          with this workflow. Otherwise, we must rely on seekpath
        - "conventional" will enforce to start the phonon computation
          from the conventional standard structure
          according to Setyawan, W., & Curtarolo, S. (2010).
          High-throughput electronic band structure calculations:
          Challenges and tools. Computational Materials Science,
          49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
          We will however use seekpath and primitive structures
          as determined by from phonopy to compute the phonon band structure
    bulk_relax_maker : .BaseVaspMaker or None
        A maker to perform a tight relaxation on the bulk.
        Set to ``None`` to skip the
        bulk relaxation
    static_energy_maker : .BaseVaspMaker or None
        A maker to perform the computation of the DFT energy on the bulk.
        Set to ``None`` to skip the
        static energy computation
    born_maker: .BaseVaspMaker or None
        Maker to compute the BORN charges.
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.
    generate_frequencies_eigenvectors_kwargs : dict
        Keyword arguments passed to :obj:`generate_frequencies_eigenvectors`.
    create_thermal_displacements: bool
        Bool that determines if thermal_displacement_matrices are computed
    kpath_scheme: str
        scheme to generate kpoints. Please be aware that
        you can only use seekpath with any kind of cell
        Otherwise, please use the standard primitive structure
        Available schemes are:
        "seekpath", "hinuma", "setyawan_curtarolo", "latimer_munro".
        "seekpath" and "hinuma" are the same definition but
        seekpath can be used with any kind of unit cell as
        it relies on phonopy to handle the relationship
        to the primitive cell and not pymatgen
    code: str
        determines the dft code. currently only vasp is implemented.
        This keyword might enable the implementation of other codes
        in the future
    store_force_constants: bool
        if True, force constants will be stored
    """

    name: str = "phonon"
    sym_reduce: bool = True
    symprec: float = 1e-4
    displacement: float = 0.01
    min_length: float | None = 20.0
    prefer_90_degrees: bool = True
    get_supercell_size_kwargs: dict = field(default_factory=dict)
    use_symmetrized_structure: str | None = None
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    static_energy_maker: BaseVaspMaker | None = field(default_factory=StaticMaker)
    born_maker: BaseVaspMaker | None = field(default_factory=DielectricMaker)
    phonon_displacement_maker: BaseVaspMaker = field(
        default_factory=PhononDisplacementMaker
    )
    create_thermal_displacements: bool = True
    generate_frequencies_eigenvectors_kwargs: dict = field(default_factory=dict)
    kpath_scheme: str = "seekpath"
    code: str = "vasp"
    store_force_constants: bool = True

    def make(
        self,
        structure: Structure,
        prev_vasp_dir: str | Path | None = None,
        born: List[Matrix3D] | None = None,
        epsilon_static: Matrix3D | None = None,
        total_dft_energy_per_formula_unit: float | None = None,
        supercell_matrix: Matrix3D | None = None,
    ):
        """
        Make flow to calculate the phonon properties.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure. Please start with a structure
            that is nearly fully optimized as the internal optimizers
            have very strict settings!
        prev_vasp_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.
        born: Matrix3D
            Instead of recomputing born charges and epsilon,
            these values can also be provided manually.
            if born, epsilon_static are provided, the born
            run will be skipped
            this matrix can be provided in the phonopy convention
            with born charges for symmetrically
            inequivalent atoms only or
            it can be provided in the VASP convention with information for
            every atom in unit cell. Please be careful when converting
            structures within in this workflow as this could lead to errors
        epsilon_static: Matrix3D
            The high-frequency dielectric constant
            Instead of recomputing born charges and epsilon,
            these values can also be provided.
            if born, epsilon_static are provided, the born
            run will be skipped
        total_dft_energy_per_formula_unit: float
            It has to be given per formula unit (as a result in corresponding Doc)
            Instead of recomputing the energy of the bulk structure every time,
            this value can also be provided in eV. If it is provided,
            the static run will be skipped. This energy is the typical
            output dft energy of the dft workflow. No conversion needed.
        supercell_matrix: list
            instead of min_length, also a supercell_matrix can
            be given, e.g. [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]
        """
        if self.use_symmetrized_structure not in [None, "primitive", "conventional"]:
            raise ValueError(
                "use_symmetrized_structure can only be primitive, conventional, None"
            )

        if (
            not self.use_symmetrized_structure == "primitive"
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
        ]:
            raise ValueError("kpath scheme is not implemented")

        jobs = []

        # TODO: should this be after or before structural
        # optimization as the optimization could change
        # the symmetry
        # we could add a tutorial and point out that the structure
        # should be nearly optimized before the phonon workflow
        if self.use_symmetrized_structure == "primitive":
            # These structures are compatible with many
            # of the kpath algorithms that are used for Materials Project
            prim_job = structure_to_primitive(structure, self.symprec)
            jobs.append(prim_job)
            structure = prim_job.output
        elif self.use_symmetrized_structure == "conventional":
            # it could be beneficial to use conventional
            # standard structures to arrive faster at supercells with right
            # angles
            conv_job = structure_to_conventional(structure, self.symprec)
            jobs.append(conv_job)
            structure = conv_job.output

        if self.bulk_relax_maker is not None:
            # optionally relax the structure
            bulk = self.bulk_relax_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
            jobs.append(bulk)
            structure = bulk.output.structure
            optimization_run_job_dir = bulk.output.dir_name
            optimization_run_uuid = bulk.output.uuid
        else:
            optimization_run_job_dir = None
            optimization_run_uuid = None

        # if supercell_matrix is None, supercell size will be determined
        # after relax maker to ensure that cell lengths are really larger
        # than threshold
        if supercell_matrix is None:
            supercell_job = get_supercell_size(
                structure,
                self.min_length,
                self.prefer_90_degrees,
                **self.get_supercell_size_kwargs,
            )
            jobs.append(supercell_job)
            supercell_matrix = supercell_job.output

        # get a phonon object from phonopy
        displacements = generate_phonon_displacements(
            structure=structure,
            supercell_matrix=supercell_matrix,
            displacement=self.displacement,
            sym_reduce=self.sym_reduce,
            symprec=self.symprec,
            use_symmetrized_structure=self.use_symmetrized_structure,
            kpath_scheme=self.kpath_scheme,
            code=self.code,
        )
        jobs.append(displacements)

        # perform the phonon displacement calculations
        vasp_displacement_calcs = run_phonon_displacements(
            displacements=displacements.output,
            structure=structure,
            supercell_matrix=supercell_matrix,
            phonon_maker=self.phonon_displacement_maker,
        )
        jobs.append(vasp_displacement_calcs)

        # Computation of static energy
        if (self.static_energy_maker is not None) and (
            total_dft_energy_per_formula_unit is None
        ):
            static_job = self.static_energy_maker.make(structure=structure)
            jobs.append(static_job)
            total_dft_energy = static_job.output.output.energy
            static_run_job_dir = static_job.output.dir_name
            static_run_uuid = static_job.output.uuid
        else:
            if total_dft_energy_per_formula_unit is not None:
                # to make sure that one can reuse results from Doc
                compute_total_energy_job = get_total_energy_per_cell(
                    total_dft_energy_per_formula_unit, structure
                )
                jobs.append(compute_total_energy_job)
                total_dft_energy = compute_total_energy_job.output
            else:
                total_dft_energy = None
            static_run_job_dir = None
            static_run_uuid = None

        # Computation of BORN charges
        if self.born_maker is not None and (born is None or epsilon_static is None):
            born_job = self.born_maker.make(structure)
            jobs.append(born_job)

            # I am not happy how we currently access "born" charges
            # This is very vasp specific code
            epsilon_static = born_job.output.calcs_reversed[0].output.epsilon_static
            born = born_job.output.calcs_reversed[0].output.outcar["born"]
            born_run_job_dir = born_job.output.dir_name
            born_run_uuid = born_job.output.uuid
        else:
            born_run_job_dir = None
            born_run_uuid = None

        phonon_collect = generate_frequencies_eigenvectors(
            supercell_matrix=supercell_matrix,
            displacement=self.displacement,
            sym_reduce=self.sym_reduce,
            symprec=self.symprec,
            use_symmetrized_structure=self.use_symmetrized_structure,
            kpath_scheme=self.kpath_scheme,
            code=self.code,
            structure=structure,
            displacement_data=vasp_displacement_calcs.output,
            epsilon_static=epsilon_static,
            born=born,
            total_dft_energy=total_dft_energy,
            static_run_job_dir=static_run_job_dir,
            static_run_uuid=static_run_uuid,
            born_run_job_dir=born_run_job_dir,
            born_run_uuid=born_run_uuid,
            optimization_run_job_dir=optimization_run_job_dir,
            optimization_run_uuid=optimization_run_uuid,
            create_thermal_displacements=self.create_thermal_displacements,
            store_force_constants=self.store_force_constants,
            **self.generate_frequencies_eigenvectors_kwargs,
        )

        jobs.append(phonon_collect)
        # create a flow including all jobs for a phonon computation
        flow = Flow(jobs, phonon_collect.output)
        return flow


# add a test with more than two atoms
