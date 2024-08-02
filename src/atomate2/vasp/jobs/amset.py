"""Module defining jobs for combining AMSET and VASP calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from click.testing import CliRunner
from jobflow import Flow, Response, job
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

from atomate2 import SETTINGS
from atomate2.common.files import get_zfile
from atomate2.utils.file_client import FileClient
from atomate2.utils.path import strip_hostname
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import HSEBSMaker, NonSCFMaker
from atomate2.vasp.sets.core import (
    HSEBSSetGenerator,
    HSEStaticSetGenerator,
    NonSCFSetGenerator,
    StaticSetGenerator,
)

if TYPE_CHECKING:
    from typing import Any

    from emmet.core.math import Vector3D
    from pymatgen.core import Structure

    from atomate2.vasp.sets.base import VaspInputGenerator


@dataclass
class DenseUniformMaker(NonSCFMaker):
    """
    Maker to perform a dense uniform non-self consistent field calculation.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "dense uniform"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: NonSCFSetGenerator(
            mode="uniform", reciprocal_density=1000, user_incar_settings={"LWAVE": True}
        )
    )


@dataclass
class StaticDeformationMaker(BaseVaspMaker):
    """
    Maker to perform a static calculations on structural deformations.

    The main difference to a normal static calculation is that this will write an
    explicit KPOINTS file, rather than using KSPACING. This is because all deformations
    ultimately need to be on exactly the same k-point mesh dimensions

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "static deformation"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"reciprocal_density": 100},
            user_incar_settings={"KSPACING": None},
        )
    )


@dataclass
class HSEStaticDeformationMaker(BaseVaspMaker):
    """
    Maker to perform an HSE06 static calculations on structural deformations.

    The main difference to a normal HSE06 static calculation is that this will write an
    explicit KPOINTS file, rather than using KSPACING. This is because all deformations
    ultimately need to be on exactly the same k-point mesh dimensions

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "static deformation"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: HSEStaticSetGenerator(
            user_kpoints_settings={"reciprocal_density": 100},
            user_incar_settings={"KSPACING": None},
        )
    )


@dataclass
class HSEDenseUniformMaker(HSEBSMaker):
    """
    Maker to perform a dense uniform non-self consistent field calculation.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "dense uniform"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: HSEBSSetGenerator(
            mode="uniform_dense",
            zero_weighted_reciprocal_density=1000,
            user_incar_settings={"LWAVE": True},
        )
    )


@job
def run_amset_deformations(
    structure: Structure,
    symprec: float = SETTINGS.SYMPREC,
    prev_dir: str | Path | None = None,
    static_deformation_maker: BaseVaspMaker | None = None,
) -> Response:
    """
    Run amset deformations.

    Note, this job will replace itself with N static calculations, where N is
    the number of deformations.

    Parameters
    ----------
    structure : .Structure
        A pymatgen structure.
    symprec : float
        Symmetry precision used to reduce the number of deformations. Set to None for
        no symmetry reduction.
    prev_dir : str or Path or None
        A previous VASP directory to use for copying VASP outputs.
    static_deformation_maker : .BaseVaspMaker or None
        A VaspMaker to use to generate the static deformation jobs.

    Returns
    -------
    List[str]
        The directory names of each deformation calculation.
    """
    from amset.deformation.generation import get_deformations

    if static_deformation_maker is None:
        static_deformation_maker = StaticDeformationMaker()

    deformations = get_deformations(0.005)
    if symprec is not None:
        deformations = list(symmetry_reduce(deformations, structure, symprec=symprec))

    statics = []
    outputs = []
    for idx, deformation in enumerate(deformations):
        # deform the structure
        dst = DeformStructureTransformation(deformation=deformation)
        deformed_structure = dst.apply_transformation(structure)

        # create the job
        static_job = static_deformation_maker.make(
            deformed_structure, prev_dir=prev_dir
        )
        static_job.append_name(f" {idx + 1}/{len(deformations)}")
        statics.append(static_job)

        # extract the outputs we want (only the dir name)
        outputs.append(static_job.output.dir_name)

    static_flow = Flow(statics, outputs)
    return Response(replace=static_flow)


@job
def calculate_deformation_potentials(
    bulk_dir: str,
    deformation_dirs: list[str],
    symprec: float = SETTINGS.SYMPREC,
    ibands: tuple[list[int], list[int]] = None,
) -> dict[str, str]:
    """
    Generate the deformation.h5 (containing deformation potentials) using AMSET.

    Note, this script just calls ``amset deform read``.

    Parameters
    ----------
    bulk_dir : str
        The folder containing the bulk calculation data.
    deformation_dirs : list of str
        A list of folders for each deformation.
    symprec : float
        The symmetry precision used to reduce the number of deformations. Set to None
        if no-symmetry reduction was applied.
    ibands : tuple of list of int
        Which bands to include in the deformation.h5 file. Given as a tuple of one or
        two lists (one for each spin channel). The bands indices are zero indexed.

    Returns
    -------
    dict
        A dictionary with the keys:

        - "dir_name": containing the directory where the deformation.h5 file was
          generated.
        - "log": The output log from ``amset deform read``.
    """
    from amset.tools.deformation import read
    from click.testing import CliRunner

    # convert arguments into their command line equivalents
    # note, amset expects the band indices to be 1 indexed, whereas we store them
    # as zero indexed
    symprec_str = "N" if symprec is None else str(symprec)

    # TODO: Handle host names properly
    bulk_dir = strip_hostname(bulk_dir)
    deformation_dirs = [strip_hostname(d) for d in deformation_dirs]
    args = [
        bulk_dir,
        *deformation_dirs,
        f"--symprec={symprec_str}",
    ]
    if ibands is not None:
        bands_str = ".".join(
            ",".join([str(idx + 1) for idx in band_ids]) for band_ids in ibands
        )
        args.append(f"--bands={bands_str}")

    runner = CliRunner()
    result = runner.invoke(read, args, catch_exceptions=False)

    # TODO: Store some information about the deformation potentials, e.g., values
    #   at CBM and VBM?
    return {"dir_name": str(Path.cwd()), "log": result.output}


@job
def calculate_polar_phonon_frequency(
    structure: Structure,
    frequencies: list[float],
    eigenvectors: list[Vector3D],
    born_effective_charges: list[Vector3D],
) -> dict[str, list[float]]:
    """
    Calculate the polar phonon frequency using amset.

    Parameters
    ----------
    structure : .Structure
        A pymatgen structure.
    frequencies : list of float
        The phonon mode frequencies in THz.
    eigenvectors : list of list of float
        The phonon eigenvectors.
    born_effective_charges : list of list of float
        The born effective charges.

    Returns
    -------
    dict
        A dictionary with the keys:

        - "frequency" (float): The polar phonon frequency.
        - "frequencies" (list[float]): A list of all phonon frequencies.
        - "weights" (list[float]): A list of weights for the frequencies.
    """
    from amset.tools.phonon_frequency import calculate_effective_phonon_frequency

    effective_frequency, weights = calculate_effective_phonon_frequency(
        np.array(frequencies),
        np.array(eigenvectors),
        np.array(born_effective_charges),
        structure,
    )
    return {
        "frequency": effective_frequency,
        "weights": weights.tolist(),
        "frequencies": frequencies,
    }


@job
def generate_wavefunction_coefficients(dir_name: str) -> dict[str, Any]:
    """
    Generate wavefunction.h5 file using amset.

    Parameters
    ----------
    dir_name : str
        Directory containing WAVECAR and vasprun.xml files (can be gzipped).

    Returns
    -------
    dict
        A dictionary with the keys:

        - "dir_name" (str): containing the directory where the wavefunction.h5 file was
          generated.
        - "log" (str): The output log from ``amset wave``.
        - "ibands" (Tuple[List[int], ...]): The bands included in the wavefunction.h5
          file. Given as a tuple of one or two lists (one for each spin channel).
          The bands indices are zero indexed.
    """
    from amset.tools.wavefunction import wave

    dir_name = strip_hostname(dir_name)  # TODO: Handle hostnames properly.
    fc = FileClient()
    files = fc.listdir(dir_name)
    vasprun_file = Path(dir_name) / get_zfile(files, "vasprun.xml")
    wavecar_file = Path(dir_name) / get_zfile(files, "WAVECAR")

    # wavecar can't be gzipped, so copy it to current directory and unzip it
    fc.copy(wavecar_file, wavecar_file.name)
    fc.gunzip(wavecar_file.name)

    args = ["--wavecar=WAVECAR", f"--vasprun={vasprun_file}"]
    runner = CliRunner()
    result = runner.invoke(wave, args, catch_exceptions=False)
    ibands = _extract_ibands(result.output)

    # remove WAVECAR from current directory
    fc.remove("WAVECAR")

    return {"dir_name": str(Path.cwd()), "log": result.output, "ibands": ibands}


def _extract_ibands(log: str) -> tuple[list[int], ...]:
    """
    Extract ibands from an ``amset wave`` log.

    Note in the log the band indices are 1 indexed but this function returns zero
    indexed band indices.

    Parameters
    ----------
    log : str
        The log from ``amset wave``.

    Returns
    -------
    tuple of list of int
        The bands included in the wavefunction.h5 file. Given as a tuple of one or two
        lists (one for each spin channel). The bands indices are zero indexed.
    """
    result_splits = log.split("\n")
    for i in range(len(result_splits)):
        if "Including bands" in result_splits[i]:
            # non-spin polarised result system
            min_band, max_band = result_splits[i].split()[-1].split("—")
            return (list(range(int(min_band) - 1, int(max_band))),)

        if "Including:" in result_splits[i]:
            amin_band, amax_band = result_splits[i + 1].split()[-1].split("—")
            bmin_band, bmax_band = result_splits[i + 2].split()[-1].split("—")
            aibands = list(range(int(amin_band) - 1, int(amax_band)))
            bibands = list(range(int(bmin_band) - 1, int(bmax_band)))

            if "up" in result_splits[i + 1]:
                # up listed first
                return aibands, bibands
            else:  # noqa: RET505
                # down listed first
                return bibands, aibands
    raise ValueError("Could not find ibands in log.")
