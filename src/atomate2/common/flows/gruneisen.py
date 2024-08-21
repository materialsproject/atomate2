"""Flows for calculating Grueneisen-Parameters."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2 import SETTINGS
from atomate2.common.jobs.gruneisen import (
    compute_gruneisen_param,
    run_phonon_jobs,
    shrink_expand_structure,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Structure

    from atomate2.aims.jobs.base import BaseAimsMaker
    from atomate2.common.flows.phonons import BasePhononMaker
    from atomate2.forcefields.jobs import ForceFieldRelaxMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class BaseGruneisenMaker(Maker, ABC):
    """
    Maker to calculate Grueneisen parameters with DFT/force field code and Phonopy.

    Calculate Grueneisen parameters by a finite volume change approach based on
    harmonic phonons.
    Initially, a tight structural relaxation is performed to obtain a structure without
    forces on the atoms. The optimized structure (ground state) is further expanded and
    shrunk by 1 % of its volume. Subsequently, supercells with one displaced atom are
    generated for all the three structures (ground state, expanded and shrunk volume)
    and accurate forces are computed for these structures. With the help of phonopy,
    these forces are then converted into a dynamical matrix. This dynamical matrix of
    three structures is then used as an input for the phonopy Grueneisen api
    to compute Grueneisen parameters.



    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    bulk_relax_maker: .ForceFieldRelaxMaker, .BaseAimsMaker, .BaseVaspMaker, or None
        A maker to perform an initial tight relaxation on the bulk.
    code: str
        determines the dft or force field code.
    const_vol_relax_maker: .ForceFieldRelaxMaker, .BaseAimsMaker,
        .BaseVaspMaker, or None. A maker to perform a tight relaxation
        on the expanded and shrunk structures at constant volume.
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
    mesh: tuple or float
        Mesh numbers along a, b, c axes used for Grueneisen parameter computation.
        Or an int or float to indicate a kpoint density.
    phonon_maker: .BasePhononMaker
        PhononMaker to run the phonon workflow.
    perc_vol: float
        Percent volume to shrink and expand ground state structure
    compute_gruneisen_param_kwargs: dict
        Keyword arguments passed to :obj:`compute_gruneisen_param`.
    symprec: float
        Symmetry precision for symmetry checks and phonon runs.
    """

    name: str = "Gruneisen"
    bulk_relax_maker: ForceFieldRelaxMaker | BaseVaspMaker | BaseAimsMaker | None = None
    code: str = None
    const_vol_relax_maker: ForceFieldRelaxMaker | BaseVaspMaker | BaseAimsMaker = None
    kpath_scheme: str = "seekpath"
    phonon_maker: BasePhononMaker = None
    perc_vol: float = 0.01
    mesh: tuple[float, float, float] | float = 7_000
    compute_gruneisen_param_kwargs: dict = field(default_factory=dict)
    symprec: float = SETTINGS.PHONON_SYMPREC

    def make(self, structure: Structure, prev_dir: str | Path | None = None) -> Flow:
        """
        Optimizes structure and runs phonon computations.

        Phonon computations are run for ground state, expanded and shrunk
        volume structures. Then, Grueneisen parameters are computed from
        this three phonon runs.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object. Please start with a structure
            that is nearly fully optimized as the internal optimizers
            have very strict settings!
        prev_dir : str or Path or None
            A previous calculation directory to use for copying outputs.
        """
        jobs = []  # initialize an empty list for jobs to be run

        # initialize an dict to store optimized structures
        opt_struct = dict.fromkeys(("ground", "plus", "minus"), None)
        prev_dir_dict = dict.fromkeys(("ground", "plus", "minus"), None)
        if (
            self.bulk_relax_maker is not None
        ):  # Optional job to relax the initial structure
            bulk_kwargs = {}
            if self.prev_calc_dir_argname is not None:
                bulk_kwargs[self.prev_calc_dir_argname] = prev_dir
            bulk = self.bulk_relax_maker.make(structure, **bulk_kwargs)
            jobs.append(bulk)
            opt_struct["ground"] = bulk.output.structure
            prev_dir = bulk.output.dir_name
            prev_dir_dict["ground"] = bulk.output.dir_name
        else:
            opt_struct["ground"] = structure
            prev_dir_dict["ground"] = prev_dir

        # Add job to get expanded and shrunk volume structures
        struct_dict = shrink_expand_structure(
            structure=bulk.output.structure, perc_vol=self.perc_vol
        )
        jobs.append(struct_dict)
        const_vol_relax_maker_kwargs = {}
        if self.prev_calc_dir_argname is not None:
            const_vol_relax_maker_kwargs[self.prev_calc_dir_argname] = prev_dir

        # get expanded structure
        const_vol_struct_plus = self.const_vol_relax_maker.make(
            structure=struct_dict.output["plus"], **const_vol_relax_maker_kwargs
        )
        const_vol_struct_plus.append_name(" plus")
        # add relax job at constant volume for expanded structure
        jobs.append(const_vol_struct_plus)

        opt_struct["plus"] = (
            const_vol_struct_plus.output.structure
        )  # store opt struct of expanded volume

        # get shrunk structure
        const_vol_struct_minus = self.const_vol_relax_maker.make(
            structure=struct_dict.output["minus"], **const_vol_relax_maker_kwargs
        )
        const_vol_struct_minus.append_name(" minus")
        # add relax job at constant volume for shrunk structure
        jobs.append(const_vol_struct_minus)

        opt_struct["minus"] = (
            const_vol_struct_minus.output.structure
        )  # store opt struct of expanded volume
        prev_dir_dict["plus"] = const_vol_struct_plus.output.dir_name
        prev_dir_dict["minus"] = const_vol_struct_minus.output.dir_name
        # go over a dict of prev_dir and use it in the maker
        phonon_jobs = run_phonon_jobs(
            opt_struct,
            self.phonon_maker,
            symprec=self.symprec,
            prev_calc_dir_argname=self.prev_calc_dir_argname,
            prev_dir_dict=prev_dir_dict,
        )
        jobs.append(phonon_jobs)
        # might not work well, put this into a job

        # get Gruneisen parameter from phonon runs yaml with phonopy api
        get_gru = compute_gruneisen_param(
            code=self.code,
            kpath_scheme=self.kpath_scheme,
            mesh=self.mesh,
            phonopy_yaml_paths_dict=phonon_jobs.output["phonon_yaml"],
            structure=opt_struct["ground"],
            symprec=self.symprec,
            phonon_imaginary_modes_info=phonon_jobs.output["imaginary_modes"],
            **self.compute_gruneisen_param_kwargs,
        )

        jobs.append(get_gru)

        return Flow(jobs, output=get_gru.output)

    @property
    @abstractmethod
    def prev_calc_dir_argname(self) -> str | None:
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """

    def __post_init__(self) -> None:
        """Test settings during the initialization."""
        if self.phonon_maker.bulk_relax_maker is not None:
            warnings.warn(
                "An additional bulk_relax_maker has been added "
                "to the phonon workflow. Please be aware "
                "that the volume needs to be kept fixed.",
                stacklevel=2,
            )
        if self.phonon_maker.symprec != self.symprec:
            warnings.warn(
                "You are using different symmetry precisions "
                "in the phonon makers and other parts of the "
                "Grüneisen workflow.",
                stacklevel=2,
            )
        if self.phonon_maker.static_energy_maker is not None:
            warnings.warn(
                "The static energy maker is not needed for this workflow.",
                stacklevel=2,
            )
