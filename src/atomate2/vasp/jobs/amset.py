"""Module defining jobs for combining AMSET and VASP calculations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from amset.deformation.generation import get_deformations
from amset.tools.deformation import read
from amset.tools.phonon_frequency import calculate_effective_phonon_frequency
from amset.tools.wavefunction import wave
from click.testing import CliRunner
from jobflow import Flow, Response, job
from pymatgen.core import Structure
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

from atomate2.common.file import get_zfile
from atomate2.common.schemas.math import Vector3D
from atomate2.settings import settings
from atomate2.utils.file_client import FileClient
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.base import VaspInputSetGenerator
from atomate2.vasp.sets.core import (
    HSEBSSetGenerator,
    HSEStaticSetGenerator,
    NonSCFSetGenerator,
    StaticSetGenerator,
)

__all__ = [
    "DenseUniformMaker",
    "StaticDeformationMaker",
    "HSEDenseUniformMaker",
    "HSEStaticDeformationMaker",
    "run_amset_deformations",
    "calculate_deformation_potentials",
    "calculate_polar_phonon_frequency",
    "generate_wavefunction_coefficients",
]


@dataclass
class DenseUniformMaker(BaseVaspMaker):
    """Maker to perform a dense uniform non-self consistent field calculation."""

    name: str = "dense uniform"
    input_set_generator: VaspInputSetGenerator = field(
        default_factory=lambda: NonSCFSetGenerator(
            mode="uniform", reciprocal_density=1000
        )
    )


@dataclass
class StaticDeformationMaker(BaseVaspMaker):
    """
    Maker to perform a static calculations on structural deformations.

    The main difference to a normal static calculation is that this will write an
    explicit KPOINTS file, rather than using KSPACING. This is because all deformations
    ultimately need to be on exactly the same k-point mesh dimensions
    """

    name: str = "static deformation"
    input_set_generator: VaspInputSetGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"reciprocal_density": 100},
            user_incar_settings={"KSPACING": None},
        )
    )


@dataclass
class HSEStaticDeformationMaker(BaseVaspMaker):
    """
    Maker to perform a HSE06 static calculations on structural deformations.

    The main difference to a normal HSE06 static calculation is that this will write an
    explicit KPOINTS file, rather than using KSPACING. This is because all deformations
    ultimately need to be on exactly the same k-point mesh dimensions
    """

    name: str = "static deformation"
    input_set_generator: VaspInputSetGenerator = field(
        default_factory=lambda: HSEStaticSetGenerator(
            user_kpoints_settings={"reciprocal_density": 100},
            user_incar_settings={"KSPACING": None},
        )
    )


@dataclass
class HSEDenseUniformMaker(BaseVaspMaker):
    """Maker to perform a dense uniform non-self consistent field calculation."""

    name: str = "dense uniform"
    input_set_generator: VaspInputSetGenerator = field(
        default_factory=lambda: HSEBSSetGenerator(
            mode="uniform_dense",
            zero_weighted_reciprocal_density=1000,
        )
    )


@job
def run_amset_deformations(
    structure: Structure,
    symprec: float = settings.SYMPREC,
    prev_vasp_dir: Union[str, Path] = None,
    static_deformation_maker: BaseVaspMaker = None,
):
    """
    Run amset deformations.

    Note, this job will replace itself with N static calculations, where N is
    the number of deformations.

    Parameters
    ----------
    structure
        A pymatgen structure.
    symprec
        Symmetry precision used to reduce the number of deformations. Set to None for
        no symmetry reduction.
    prev_vasp_dir
        A previous VASP directory to use for copying VASP outputs.
    static_deformation_maker
        A VaspMaker to use to generate the static deformation jobs.

    Returns
    -------
    List[str]
        The directory names of each deformation calculation.
    """
    if static_deformation_maker is None:
        static_deformation_maker = StaticDeformationMaker()

    deformations = get_deformations(0.005)
    if symprec is not None:
        deformations = list(symmetry_reduce(deformations, structure, symprec=symprec))

    statics = []
    outputs = []
    for i, deformation in enumerate(deformations):
        # deform the structure
        dst = DeformStructureTransformation(deformation=deformation)
        deformed_structure = dst.apply_transformation(structure)

        # create the job
        static_job = static_deformation_maker.make(
            deformed_structure, prev_vasp_dir=prev_vasp_dir
        )
        static_job.name += f" {i + 1}/{len(deformations)}"
        statics.append(static_job)

        # extract the outputs we want (only the dir name)
        outputs.append(static_job.dir_name)

    static_flow = Flow(statics, outputs)
    return Response(replace=static_flow)


@job
def calculate_deformation_potentials(
    bulk_dir: str,
    deformation_dirs: List[str],
    symprec: float = settings.SYMPREC,
    ibands: Tuple[List[int], List[int]] = None,
):
    """
    Generate the deformation.h5 (containing deformation potentials) using AMSET.

    Note, this script just calls ``amset deform read``.

    Parameters
    ----------
    bulk_dir
        The folder containing the bulk calculation data.
    deformation_dirs
        A list of folders for each deformation.
    symprec
        The symmetry precision used to reduce the number of deformations. Set to None
        if no-symmetry reduction was applied.
    ibands
        Which bands to include in the deformation.h5 file. Given as a tuple of one or
        two lists (one for each spin channel). The bands indices are zero indexed.

    Returns
    -------
    dict[str, str]
        A dictionary with the keys:

        - "dir_name": containing the directory where the deformation.h5 file was
          generated.
        - "log": The output log from ``amset deform read``.
    """
    from click.testing import CliRunner

    # convert arguments into their command line equivalents
    # note, amset expects the band indices to be 1 indexed, whereas we store them
    # as zero indexed
    bands_str = ".".join(",".join([str(idx + 1) for idx in b]) for b in ibands)
    symprec_str = "N" if symprec is None else str(symprec)

    args = [
        bulk_dir,
        *deformation_dirs,
        f"--bands={bands_str}",
        f"--symprec={symprec_str}",
    ]
    runner = CliRunner()
    result = runner.invoke(read, args, catch_exceptions=False)

    # TODO: Store some information about the deformation potentials, e.g., values
    #   at CBM and VBM?
    return {"dir_name": Path.cwd(), "log": result.output}


@job
def calculate_polar_phonon_frequency(
    structure: Structure,
    frequencies: List[float],
    eigenvectors: List[Vector3D],
    born_effective_charges: List[Vector3D],
):
    """
    Calculate the polar phonon frequency using amset.

    Parameters
    ----------
    structure
        A pymatgen structure.
    frequencies
        The phonon mode frequencies in THz.
    eigenvectors
        The phonon eigenvectors.
    born_effective_charges
        The born effective charges.

    Returns
    -------
    dict[str, any]
        A dictionary with the keys:

        - "frequency" (float): The polar phonon frequency.
        - "frequencies" (list[float]): A list of all phonon frequencies.
        - "weights" (list[float]): A list of weights for the frequencies.
    """
    frequencies = np.array(frequencies)
    eigenvectors = np.array(eigenvectors)
    born_effective_charges = np.array(born_effective_charges)

    effective_frequency, weights = calculate_effective_phonon_frequency(
        frequencies, eigenvectors, born_effective_charges, structure
    )
    return {
        "frequency": effective_frequency,
        "weights": weights,
        "frequencies": frequencies,
    }


@job
def generate_wavefunction_coefficients(dir_name: str):
    """
    Generate wavefunction.h5 file using amset.

    Parameters
    ----------
    dir_name
        Directory containing WAVECAR and vasprun.xml files (can be gzipped).

    Returns
    -------
    dict[str, str]
        A dictionary with the keys:

        - "dir_name" (str): containing the directory where the wavefunction.h5 file was
          generated.
        - "log" (str): The output log from ``amset wave``.
        - "ibands" (Tuple[List[int], ...]): The bands included in the wavefunction.h5
          file. Given as a tuple of one or two lists (one for each spin channel).
          The bands indices are zero indexed.
    """
    files = FileClient().listdir(dir_name)
    vasprun_file = get_zfile(files, "vasprun.xml")
    wavecar_file = get_zfile(files, "WAVECAR")

    args = [
        f"--wavecar={wavecar_file}",
        f"--vasprun={vasprun_file}",
    ]
    runner = CliRunner()
    result = runner.invoke(wave, args, catch_exceptions=False)
    ibands = _extract_ibands(result.output)

    return {"dir_name": Path.cwd(), "log": result.output, "ibands": ibands}


def _extract_ibands(log) -> Tuple[List[int], ...]:
    """
    Extract ibands from an ``amset wave`` log.

    Note in the log the band indices are 1 indexed but this function returns zero
    indexed band indices.

    Parameters
    ----------
    log
        The log from ``amset wave``.

    Returns
    -------
    tuple[list[int], ...]
        The bands included in the wavefunction.h5 file. Given as a tuple of one or two
        lists (one for each spin channel). The bands indices are zero indexed.
    """
    result_splits = log.output.split("\n")
    for i in range(len(result_splits)):
        if "Including bands" in result_splits[i]:
            # non-spin polarised result system
            min_band, max_band = result_splits[i].split(" ")[-1].split("-")
            return (list(range(int(min_band) - 1, int(max_band))),)

        if "Including:" in result_splits[i]:
            amin_band, amax_band = result_splits[i + 1].split(" ")[-1].split("-")
            bmin_band, bmax_band = result_splits[i + 2].split(" ")[-1].split("-")
            aibands = list(range(int(amin_band) - 1, int(amax_band)))
            bibands = list(range(int(bmin_band) - 1, int(bmax_band)))

            if "up" in result_splits[i + 1]:
                # up listed first
                return aibands, bibands
            else:
                # down listed first
                return bibands, aibands
    raise ValueError("Could not find ibands in log.")
