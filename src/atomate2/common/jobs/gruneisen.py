"""Jobs for Grueneisen parameter computations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jobflow import Flow, Response, job
from pymatgen.phonon.gruneisen import (
    GruneisenParameter,
    GruneisenPhononBandStructureSymmLine,
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2 import SETTINGS
from atomate2.common.schemas.gruneisen import GruneisenParameterDocument
from atomate2.common.schemas.phonons import PhononBSDOSDoc

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

    from atomate2.common.flows.phonons import BasePhononMaker

logger = logging.getLogger(__name__)


@job
def shrink_expand_structure(structure: Structure, perc_vol: float) -> Response:
    """Create structures with expanded and reduced volumes.

    Parameters
    ----------
    structure: .Structure
        optimized pymatgen structure obj
    perc_vol: tuple(float, float)
        percentage to shrink and expand the volume

    Returns
    -------
    Response object with structures that have expanded and reduced volumes
    """
    plus_struct = structure.copy()
    minus_struct = structure.copy()

    plus_struct.scale_lattice(volume=structure.volume * (1 + perc_vol))
    minus_struct.scale_lattice(volume=structure.volume * (1 - perc_vol))

    return Response(output={"plus": plus_struct, "minus": minus_struct})


@job(data=[PhononBSDOSDoc])
def run_phonon_jobs(
    opt_struct: dict,
    phonon_maker: BasePhononMaker = None,
    symprec: float = SETTINGS.SYMPREC,
    prev_calc_dir_argname: str = None,
    prev_dir_dict: dict = None,
) -> Response:
    """Run all phonon jobs if the symmetry stayed the same.

    Parameters
    ----------
    opt_struct: dict
        including all optimized structures with the keys ground, plus, minus
    phonon_maker: .BasePhononMaker
        Maker to run a harmonic phonon computation.
    symprec: float
        symmetry precision for phonon computation.
    prev_calc_dir_argname: str or None
        name of the argument for the previous calculation directory
    prev_dir_dict: dict
        dictionary of previous calculation directories keyed by the different
        types of optimization runs

    Returns
    -------
    Phonon Jobs or Symmetry of the optimized structures.
    """
    symmetry = []
    for st in opt_struct:
        sga = SpacegroupAnalyzer(opt_struct[st], symprec=symprec)
        symmetry.append(int(sga.get_space_group_number()))
    set_symmetry = list(set(symmetry))
    if len(set_symmetry) == 1:
        jobs = []
        phonon_yaml_dirs = dict.fromkeys(("ground", "plus", "minus"), None)
        phonon_imaginary_modes = dict.fromkeys(("ground", "plus", "minus"), None)
        for st in opt_struct:
            # phonon run for all 3 optimized structures (ground state, expanded, shrunk)
            phonon_kwargs = {}
            if prev_calc_dir_argname is not None:
                phonon_kwargs[prev_calc_dir_argname] = prev_dir_dict[st]
            phonon_job = phonon_maker.make(structure=opt_struct[st], **phonon_kwargs)
            phonon_job.append_name(f" {st}")
            # change default phonopy.yaml file name to ensure workflow can be
            # run without having to create folders, thus
            # prevent overwriting and easier to identify yaml file belong
            # to corresponding phonon run
            phonon_job.jobs[-1].function_kwargs.update(
                filename_phonopy_yaml=f"{st}_phonopy.yaml",
                filename_band_yaml=f"{st}_phonon_band_structure.yaml",
                filename_dos_yaml=f"{st}_phonon_dos.yaml",
                filename_bs=f"{st}_phonon_band_structure.pdf",
                filename_dos=f"{st}_phonon_dos.pdf",
            )
            jobs.append(phonon_job)
            # store each phonon run task doc
            phonon_yaml_dirs[st] = phonon_job.output.jobdirs.taskdoc_run_job_dir
            phonon_imaginary_modes[st] = phonon_job.output.has_imaginary_modes

        return Response(
            replace=Flow(jobs),
            output={
                "phonon_yaml": phonon_yaml_dirs,
                "imaginary_modes": phonon_imaginary_modes,
            },
        )
    logger.warning(
        msg="Different space groups were detected for the optimized structures."
        "Please try a different symprec."
    )
    return Response(output={"error": "different space groups"}, stop_jobflow=True)


@job(
    output_schema=GruneisenParameterDocument,
    data=[GruneisenParameter, GruneisenPhononBandStructureSymmLine],
)
def compute_gruneisen_param(
    code: str,
    phonopy_yaml_paths_dict: dict,
    phonon_imaginary_modes_info: dict,
    kpath_scheme: str,
    symprec: float,
    mesh: tuple | float = (20, 20, 20),
    structure: Structure = None,
    **compute_gruneisen_param_kwargs,
) -> GruneisenParameterDocument:
    """Compute Grueneisen parameters from phonon runs.

    Requires phonopy yaml files from ground, expanded and contracted structures.

    Parameters
    ----------
    code: str
        Code to compute forces
    phonopy_yaml_paths_dict:
        phonopy yaml files path for ground, expanded and
        contracted structure phonon runs
    phonon_imaginary_modes_info:
        dict with bool indicating if structure
        has imaginary modes
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
    symprec: float
        Symmetry precision for symmetry checks and phonon runs.
    mesh: float or int or tuple(int, int, int)
        kpoint density (float, int) or sampling mesh (tuple(int, int, int))
    structure: .Structure
        pymatgen structure object at ground state
    compute_gruneisen_param_kwargs:
        kwargs for phonopy Grueneisen
        api and pymatgen plotters

    Returns
    -------
    .GruneisenParameterDocument
    """
    return GruneisenParameterDocument.from_phonon_yamls(
        code=code,
        compute_gruneisen_param_kwargs=compute_gruneisen_param_kwargs,
        kpath_scheme=kpath_scheme,
        mesh=mesh,
        phonon_imaginary_modes_info=phonon_imaginary_modes_info,
        phonopy_yaml_paths_dict=phonopy_yaml_paths_dict,
        structure=structure,
        symprec=symprec,
    )
